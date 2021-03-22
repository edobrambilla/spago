// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trainer

import (
	"bufio"
	"fmt"
	"github.com/gosuri/uiprogress"
	"github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/floatutils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/stats"
	"io"
	"os"
)

type Evaluator struct {
	model          nn.Model
	modelType      string
	evalCorpusPath string
	includeTitle   bool
	includeBody    bool
	countLines     int
	labelsMap      map[string]int
}

func NewEvaluator(model nn.Model, t TrainingConfig, modelType string) *Evaluator {
	e := &Evaluator{
		model:          model,
		modelType:      modelType,
		evalCorpusPath: t.EvalCorpusPath,
		includeTitle:   t.IncludeTitle,
		includeBody:    t.IncludeBody,
		labelsMap:      t.LabelsMap,
	}
	err := e.forEachLine(func(i int, text string) {
		e.countLines++
	})
	if err != nil && err != io.EOF {
		fmt.Printf(" > Failed with error: %v\n", err)
	}
	return e
}

// Predict performs the forward pass and returns the predict label
func (e *Evaluator) Predict(tokenizedExample []string) int {
	g := ag.NewGraph()
	c := nn.Context{Graph: g, Mode: nn.Inference}
	var y ag.Node
	var category int
	if e.modelType == "birnn" {
		model := nn.Reify(c, e.model).(*BiRNNClassifierModel)
		y = model.Forward(tokenizedExample)[0]
		category = floatutils.ArgMax(y.Value().Data())
		//println (category)
	} else if e.modelType == "xdnn" {
		model := nn.Reify(c, e.model).(*BiRNNClassifierModel)
		encodedText := model.EncodeText(tokenizedExample)
		category = model.XDNNModel.Classify(encodedText.Value().(*mat32.Dense))
		//println (category)
	} else {
		panic("Evaluator: Wrong model type")
	}
	defer g.Clear()
	return category
}

func (e *Evaluator) Evaluate(epoch int) *stats.ClassMetrics {
	uip := uiprogress.New()
	bar := e.newTestBar(uip, e.countLines, epoch)
	uip.Start()

	counter := stats.NewMetricCounter()
	err := e.forEachLine(func(i int, text string) {
		//t.trainPassage(text)
		example := GetExample(text)
		tokenizedExample := GetTokenizedExample(example, e.includeTitle, e.includeBody)
		if len(tokenizedExample) > 0 {
			tokenizedExample = PadTokens(tokenizedExample, 5)
			println("---------")
			println("category " + string(e.labelsMap[example.Category]))
			if e.Predict(tokenizedExample) == e.labelsMap[example.Category] {
				counter.IncTruePos()
			} else {
				counter.IncFalsePos()
			}
			bar.Incr()
		}
	})
	uip.Stop()
	if err != nil && err != io.EOF {
		fmt.Printf(" > Failed with error: %v\n", err)
	}
	return counter
}

func (e *Evaluator) forEachLine(callback func(i int, line string)) (err error) {
	file, err := os.Open(e.evalCorpusPath)
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

func (e *Evaluator) newTestBar(progress *uiprogress.Progress, nexamples, curEpoch int) *uiprogress.Bar {
	bar := progress.AddBar(nexamples)
	bar.AppendCompleted().PrependElapsed()
	bar.PrependFunc(func(b *uiprogress.Bar) string {
		return fmt.Sprintf("Epoch: %d Evaluation", curEpoch)
	})
	return bar
}
