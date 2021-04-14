// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trainer

import (
	"bufio"
	"fmt"
	"github.com/gosuri/uiprogress"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/losses"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"github.com/nlpodyssey/spago/pkg/nlp/classifiers/prado"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
	"github.com/nlpodyssey/spago/pkg/utils"
	"io"
	"os"
)

type PradoTrainer struct {
	TrainingConfig
	randGen       *rand.LockedRand
	optimizer     *gd.GradientDescent
	bestLoss      float32
	lastBatchLoss float32
	model         *prado.Model
	countLines    int
	curLoss       float32
	curEpoch      int
}

func NewPradoTrainer(model *prado.Model, config TrainingConfig, optimizer *gd.GradientDescent) *PradoTrainer {
	return &PradoTrainer{
		TrainingConfig: config,
		randGen:        rand.NewLockedRand(config.Seed),
		optimizer:      optimizer,
		model:          model,
	}
}

func (t *PradoTrainer) GetVocabulary() *vocabulary.Vocabulary {
	out := vocabulary.New([]string{})
	err := t.forEachLine(func(i int, text string) {
		e := GetExample(text)
		tokenizedExample := GetTokenizedExample(e, t.TrainingConfig.IncludeTitle, t.TrainingConfig.IncludeBody)
		tokenizedExample = PadTokens(tokenizedExample, 5)
		for _, word := range tokenizedExample {
			out.Add(word)
		}
		t.countLines++
	})
	if err != nil && err != io.EOF {
		fmt.Printf(" > Failed with error: %v\n", err)
	}
	return out
}

func (t *PradoTrainer) trainBatches(onExample func()) {
	batches := make([][]string, 0)
	labels := make([]int, 0)
	b := 0
	g := ag.NewGraph(ag.Rand(rand.NewLockedRand(t.Seed)))
	c := nn.Context{Graph: g, Mode: nn.Training}
	model := nn.Reify(c, t.model).(*prado.Model)

	err := t.forEachLine(func(i int, text string) {

		e := GetExample(text)
		tokenizedExample := GetTokenizedExample(e, t.TrainingConfig.IncludeTitle, t.TrainingConfig.IncludeBody)
		if len(tokenizedExample) > 0 {
			tokenizedExample = PadTokens(tokenizedExample, 5)
			batches = append(batches, tokenizedExample)
			labels = append(labels, t.TrainingConfig.LabelsMap[e.Category])
			t.optimizer.IncExample()

			if (b%t.TrainingConfig.BatchSize == 0) || i == t.countLines {
				t.curLoss = t.trainBatch(model, batches, labels)
				t.optimizer.IncBatch()
				t.optimizer.Optimize()
				batches = make([][]string, 0)
				labels = make([]int, 0)
			}
		}
		b += 1
		onExample()
	})
	if err != nil && err != io.EOF {
		fmt.Printf(" > Failed with error: %v\n", err)
	}
}

// trainbatch performs the backward respect to the cross-entropy loss, returned as scalar value
func (t *PradoTrainer) trainBatch(model *prado.Model, batch [][]string, labels []int) float32 {
	g := model.Graph()
	defer g.Clear()
	var loss ag.Node
	for e := 0; e < len(batch); e++ {
		y := model.Forward(batch[e])[0]
		label := labels[e]
		loss = g.Add(loss, losses.FocalLoss(g, y, label, 2.0))
	}
	//loss = g.Div(loss, g.NewScalar(float32(t.BatchSize)))
	g.Backward(loss)
	return loss.ScalarValue()
}

func (t *PradoTrainer) forEachLine(callback func(i int, line string)) (err error) {
	file, err := os.Open(t.TrainCorpusPath)
	if err != nil {
		return err
	}
	defer file.Close()

	// Start reading from the file with a reader.
	reader := bufio.NewReader(file)
	var line string
	i := 0
	for {
		line, err = reader.ReadString('\n')
		if err != nil && err != io.EOF {
			break
		}
		// Process the line here
		callback(i, line)
		i++
		if err != nil {
			break
		}
	}
	if err != io.EOF {
		fmt.Printf(" > Failed with error: %v\n", err)
		return err
	}
	return
}

func (t *PradoTrainer) newTrainBar(progress *uiprogress.Progress, nexamples int) *uiprogress.Bar {
	bar := progress.AddBar(nexamples)
	bar.AppendCompleted().PrependElapsed()
	bar.PrependFunc(func(b *uiprogress.Bar) string {
		return fmt.Sprintf("Epoch: %d Loss: %.6f", t.curEpoch, t.curLoss)
	})
	return bar
}

func (t *PradoTrainer) Enjoy() {
	for epoch := 1; epoch <= t.Epochs; epoch++ {
		t.curEpoch = epoch
		t.optimizer.IncEpoch()

		fmt.Println("Training epoch...")
		t.trainEpoch()

		// model serialization
		err := utils.SerializeToFile(t.ModelPath, t.model)
		if err != nil {
			panic("mnist: error during model serialization.")
		}
	}
}

func (t *PradoTrainer) trainEpoch() {
	uip := uiprogress.New()
	bar := t.newTrainBar(uip, t.countLines)
	uip.Start() // start bar rendering
	t.trainBatches(func() { bar.Incr() })
	uip.Stop()
	precision := NewEvaluator(t.model, t.TrainingConfig, "prado").Evaluate(t.curEpoch).Precision()
	fmt.Printf("Accuracy: %.2f\n", 100*precision)
}
