// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/nlpodyssey/spago/examples/text_classifier/trainer"
	"github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
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

func pradoTrain(modelPath string, trainingPath string, testPath string) {
	model := newPradoModel()
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
	t := trainer.NewPradoTrainer(model, config, optimizer)
	//get vocabulary
	v := t.GetVocabulary()
	vocabularyCodes := getHashedVocabulary(v, model.Embeddings.EmbeddingsConfig)
	model.Vocabulary = v
	model.Embeddings.SetProjectedEmbeddings(vocabularyCodes)
	println("Corpus examples: " + string(v.Size()))
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
	pradoTrain(modelPath, trainingPath, testPath)

}

func newPradoModel() *prado.Model {
	config := prado.Config{
		EncodingActivation:    "ReLU",
		ConvActivation:        "Identity",
		ConvSize:              4,
		InputSize:             30,
		ProjectionSize:        128,
		ProjectionArity:       3,
		EncodingSize:          96,
		UnigramsChannels:      1,
		BigramsChannels:       1,
		TrigramsChannels:      1,
		FourgramsChannels:     0,
		FivegramsChannels:     0,
		Skip1BigramsChannels:  1,
		Skip2BigramsChannels:  1,
		Skip1TrigramsChannels: 1,
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

func getHashCode(config prado.EmbeddingsConfig, r *rand.LockedRand) mat32.Matrix {
	out := mat32.NewEmptyVecDense(config.InputSize)
	c := 0
	for i := 0; i < config.InputSize; i++ {
		out.Data()[c] = (r.Float32() * 2.0) - 1.0
		c++
	}
	return out //.ProdScalar(0.1)
}

func getStringCode(s string, config prado.EmbeddingsConfig) mat32.Matrix {
	out := mat32.NewEmptyVecDense(config.InputSize)
	c := 0
	for _, char := range s {
		if c < config.InputSize {
			for n := 1; n <= 3; n++ {
				out.Data()[c] = mat32.Float(float64(digit(int(char), n)))
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

func getHashedVocabulary(vocabulary *vocabulary.Vocabulary, config prado.EmbeddingsConfig) map[string]mat32.Matrix {
	var outMap map[string]mat32.Matrix
	outMap = make(map[string]mat32.Matrix)
	//r := rand.NewLockedRand(40)
	for _, word := range vocabulary.Items() {
		outMap[word] = getStringCode(word, config)
	}
	return outMap
}
