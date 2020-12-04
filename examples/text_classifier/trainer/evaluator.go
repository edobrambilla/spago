// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trainer

import (
	"bufio"
	"fmt"
	"github.com/gosuri/uiprogress"
	"github.com/nlpodyssey/spago/pkg/mat/f64utils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/stats"
	"github.com/nlpodyssey/spago/pkg/nlp/classifiers/prado"
	"io"
	"os"
)

type Evaluator struct {
	model          nn.Model
	evalCorpusPath string
	includeTitle   bool
	includeBody    bool
	labelsMap      map[string]int
}

func NewEvaluator(model nn.Model, t TrainingConfig) *Evaluator {
	e := &Evaluator{
		model:          model,
		evalCorpusPath: t.EvalCorpusPath,
		includeTitle:   t.IncludeTitle,
		includeBody:    t.IncludeBody,
		labelsMap:      t.LabelsMap,
	}
	return e
}

// Predict performs the forward pass and returns the predict label
func (e *Evaluator) Predict(tokenizedExample []string) int {
	g := ag.NewGraph()
	defer g.Clear()
	proc := e.model.NewProc(g)
	proc.SetMode(nn.Inference) // Important
	y := proc.(*prado.Processor).Classify(tokenizedExample)[0]
	return f64utils.ArgMax(y.Value().Data())
}

func (e *Evaluator) Evaluate() *stats.ClassMetrics {
	uip := uiprogress.New()
	bar := newTestBar(uip, len(e.evalCorpusPath))
	uip.Start()
	defer uip.Stop()

	counter := stats.NewMetricCounter()
	err := e.forEachLine(func(i int, text string) {
		//t.trainPassage(text)
		example := GetExample(text)
		tokenizedExample := GetTokenizedExample(example, e.includeTitle, e.includeBody)
		if e.Predict(tokenizedExample) == e.labelsMap[example.Category] {
			counter.IncTruePos()
		} else {
			counter.IncFalsePos()
		}
		bar.Incr()
	})
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

func newTestBar(p *uiprogress.Progress, nexamples int) *uiprogress.Bar {
	bar := p.AddBar(nexamples)
	bar.AppendCompleted().PrependElapsed()
	return bar
}
