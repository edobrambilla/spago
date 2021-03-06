// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/nlpodyssey/spago/pkg/examples/text_classifier/xdnn_classifier"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd/adam"
	"log"
	"net/http"
	"os"
)

func biRNNTrain(modelPath string, trainingPath string, testPath string) {
	model := newBiRNNModel()
	model.InitBiRNNParameters(rand.NewLockedRand(743))
	updater := adam.New(adam.NewDefaultConfig())
	optimizer := gd.NewOptimizer(updater, nn.NewDefaultParamsIterator(model))
	config := trainer.TrainingConfig{
		Seed:             743,
		BatchSize:        1,
		Epochs:           1,
		GradientClipping: 0,
		TrainCorpusPath:  trainingPath,
		EvalCorpusPath:   testPath,
		ModelPath:        modelPath,
		IncludeBody:      false,
		IncludeTitle:     true,
		LabelsMap: map[string]int{
			"01000000": 0,
			"02000000": 1,
			"03000000": 2,
			"04000000": 3,
			"05000000": 4,
			"06000000": 5},
	}
	t := trainer.NewBiRNNTrainer(model, config, optimizer)
	//get vocabulary
	v := t.GetVocabulary()
	model.SetEmbeddings(*v)
	print("Vocabulary size: ")
	println(v.Size())
	t.Enjoy()
}

func main() {
	// go tool pprof http://localhost:6060/debug/pprof/profile
	go func() { log.Println(http.ListenAndServe("localhost:6060", nil)) }()

	modelPath := os.Args[1]
	var trainingPath string
	var testPath string
	if len(os.Args) > 2 {
		trainingPath = os.Args[2]
	} else {
		panic("Undefined training corpus path")
	}
	if len(os.Args) > 3 {
		testPath = os.Args[3]
	}

	biRNNTrain(modelPath, trainingPath, testPath)
}

func newBiRNNModel() *trainer.BiRNNClassifierModel {
	config := trainer.BiRNNClassifierConfig{
		EncodingActivation: "ReLU",
		InputSize:          100,
		OutputSize:         100,
		TypeVocabSize:      0,
		VocabSize:          5,
		Id2Label: map[string]string{
			"0": "01000000",
			"1": "02000000",
			"2": "03000000",
			"3": "04000000",
			"4": "05000000",
			"5": "06000000",
		},
	}
	return trainer.NewBiRNNClassifierModel(config)
}
