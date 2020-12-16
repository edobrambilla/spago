// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/nlpodyssey/spago/examples/text_classifier/trainer"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd/adam"
	"github.com/nlpodyssey/spago/pkg/nlp/classifiers/prado"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
	"log"
	"math"
	"net/http"
	"os"
)

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
	model := newTestModel()
	model.InitPradoParameters(rand.NewLockedRand(743))
	updater := adam.New(adam.NewDefaultConfig())
	optimizer := gd.NewOptimizer(updater, nn.NewDefaultParamsIterator(model))
	config := trainer.TrainingConfig{
		Seed:             743,
		BatchSize:        1,
		Epochs:           8,
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
	t := trainer.NewTrainer(model, config, optimizer)
	//get vocabulary
	v := t.GetVocabulary()
	vocabularyCodes := getHashedVocabulary(v)
	model.Vocabulary = v
	model.Embeddings.SetProjectedEmbeddings(vocabularyCodes)
	print(v.Size())
	// read dataset
	t.Enjoy()

}

func newTestModel() *prado.Model {
	config := prado.Config{
		EncodingActivation:    "Tanh",
		ConvActivation:        "Tanh",
		ConvSize:              4,
		InputSize:             30,
		ProjectionSize:        128,
		ProjectionArity:       3,
		EncodingSize:          36,
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
			"0": "01000000",
			"1": "02000000",
			"2": "03000000",
			"3": "04000000",
			"4": "05000000",
			"5": "06000000",
		},
	}
	return prado.NewDefaultPrado(config, "path")
}

func getStringCode(s string) mat.Matrix {
	out := mat.NewEmptyVecDense(30)
	c := 0
	for _, char := range s {
		if c < 30 {
			for n := 1; n <= 3; n++ {
				out.Data()[c] = float64(digit(int(char), n))
				c++
			}
		}
	}
	return out.ProdScalar(0.1)
}

func digit(num, place int) int {
	r := num % int(math.Pow(10, float64(place)))
	return r / int(math.Pow(10, float64(place-1)))
}

func getHashedVocabulary(vocabulary *vocabulary.Vocabulary) map[string]mat.Matrix {
	var outMap map[string]mat.Matrix
	outMap = make(map[string]mat.Matrix)
	for _, word := range vocabulary.Items() {
		outMap[word] = getStringCode(word)
	}
	return outMap
}
