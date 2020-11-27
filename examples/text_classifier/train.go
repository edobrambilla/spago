// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/nlpodyssey/spago/examples/text_classifier/trainer"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd/adam"
	"github.com/nlpodyssey/spago/pkg/nlp/classifiers/prado"
	"log"
	"net/http"
	"os"
)

func main() {
	// go tool pprof http://localhost:6060/debug/pprof/profile
	go func() { log.Println(http.ListenAndServe("localhost:6060", nil)) }()

	modelPath := os.Args[1]
	var datasetPath string
	if len(os.Args) > 2 {
		datasetPath = os.Args[2]
	} else {
		panic("Undefined corpus path")
	}
	model := newTestModel()
	updater := adam.New(adam.NewDefaultConfig())
	optimizer := gd.NewOptimizer(updater, nn.NewDefaultParamsIterator(model))
	config := trainer.TrainingConfig{
		Seed:             743,
		BatchSize:        1,
		Epochs:           2,
		GradientClipping: 0,
		TrainCorpusPath:  datasetPath,
		ModelPath:        modelPath,
	}
	t := trainer.NewTrainer(model, config, optimizer)

	// read dataset
	t.Train()

}

func newTestModel() *prado.Model {
	config := prado.Config{
		EncodingActivation:    "Identity",
		ConvActivation:        "Tanh",
		ConvSize:              4,
		InputSize:             30,
		ProjectionSize:        128,
		ProjectionArity:       3,
		EncodingSize:          32,
		UnigramsChannels:      1,
		BigramsChannels:       1,
		TrigramsChannels:      1,
		FourgramsChannels:     0,
		FivegramsChannels:     0,
		Skip1BigramsChannels:  1,
		Skip2BigramsChannels:  0,
		Skip1TrigramsChannels: 0,
		TypeVocabSize:         0,
		VocabSize:             5,
		Id2Label: map[string]string{
			"1": "01000000",
			"2": "01000000",
			"3": "01000000",
			"4": "01000000",
			"5": "01000000",
			"6": "01000000",
		},
	}
	return prado.NewDefaultPrado(config, "path")
}
