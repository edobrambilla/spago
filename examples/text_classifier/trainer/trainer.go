// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trainer

import (
	"bufio"
	"fmt"
	"github.com/gosuri/uiprogress"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
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

type TrainingConfig struct {
	Seed             uint64
	BatchSize        int
	Epochs           int
	GradientClipping float64
	TrainCorpusPath  string
	EvalCorpusPath   string
	ModelPath        string
	IncludeTitle     bool
	IncludeBody      bool
	LabelsMap        map[string]int
}

type Trainer struct {
	TrainingConfig
	randGen       *rand.LockedRand
	optimizer     *gd.GradientDescent
	bestLoss      float64
	lastBatchLoss float64
	model         *prado.Model
	countLines    int
	curLoss       float64
	curEpoch      int
}

func NewTrainer(model *prado.Model, config TrainingConfig, optimizer *gd.GradientDescent) *Trainer {
	return &Trainer{
		TrainingConfig: config,
		randGen:        rand.NewLockedRand(config.Seed),
		optimizer:      optimizer,
		model:          model,
	}
}

func (t *Trainer) GetVocabulary() *vocabulary.Vocabulary {
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

func (t *Trainer) trainBatches(onExample func()) {
	err := t.forEachLine(func(i int, text string) {
		//t.trainPassage(text)
		e := GetExample(text)
		tokenizedExample := GetTokenizedExample(e, t.TrainingConfig.IncludeTitle, t.TrainingConfig.IncludeBody)
		if len(tokenizedExample) > 0 {
			tokenizedExample = PadTokens(tokenizedExample, 5)
			t.curLoss = t.learn(i, tokenizedExample, t.TrainingConfig.LabelsMap[e.Category])
			t.optimizer.IncBatch()
			t.optimizer.IncExample()
			t.optimizer.Optimize()
		}
		onExample()
	})
	if err != nil && err != io.EOF {
		fmt.Printf(" > Failed with error: %v\n", err)
	}
}

// learn performs the backward respect to the cross-entropy loss, returned as scalar value
func (t *Trainer) learn(_ int, tokenizedExample []string, label int) float64 {
	g := ag.NewGraph(ag.Rand(rand.NewLockedRand(t.Seed)))
	//g := ag.NewGraph(ag.Rand(rand.NewLockedRand(t.Seed)), ag.IncrementalForward(false), ag.ConcurrentComputations(true))
	defer g.Clear()
	y := t.model.NewProc(g).(*prado.Processor).Classify(tokenizedExample)[0]
	loss := g.Div(losses.CrossEntropy(g, y, label), g.NewScalar(1.0))
	//g.Forward()
	g.Backward(loss)
	return loss.ScalarValue()
}

func (t *Trainer) forEachLine(callback func(i int, line string)) (err error) {
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

func (t *Trainer) newTrainBar(progress *uiprogress.Progress, nexamples int) *uiprogress.Bar {
	bar := progress.AddBar(nexamples)
	bar.AppendCompleted().PrependElapsed()
	bar.PrependFunc(func(b *uiprogress.Bar) string {
		return fmt.Sprintf("Epoch: %d Loss: %.6f", t.curEpoch, t.curLoss)
	})
	return bar
}

func (t *Trainer) Enjoy() {
	for epoch := 0; epoch < t.Epochs; epoch++ {
		t.curEpoch = epoch
		t.optimizer.IncEpoch()

		fmt.Println("Training epoch...")
		t.trainEpoch()

		// evaluate here

		// model serialization
		err := utils.SerializeToFile(t.ModelPath, nn.NewParamsSerializer(t.model))
		if err != nil {
			panic("mnist: error during model serialization.")
		}
	}
}

func (t *Trainer) trainEpoch() {
	uip := uiprogress.New()
	bar := t.newTrainBar(uip, t.countLines)
	uip.Start() // start bar rendering
	t.trainBatches(func() { bar.Incr() })
	uip.Stop()
	precision := NewEvaluator(t.model, t.TrainingConfig).Evaluate(t.curEpoch).Precision()
	fmt.Printf("Accuracy: %.2f\n", 100*precision)
}
