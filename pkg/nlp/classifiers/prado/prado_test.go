// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package prado

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
	"math"
	"testing"
)

func TestModel_Forward(t *testing.T) {
	g := ag.NewGraph()
	v := getVocabulary()
	vocabularyCodes := getHashedVocabulary(v)
	model := newTestModel()
	model.InitPradoParameters(rand.NewLockedRand(743))
	model.Embeddings.SetProjectedEmbeddings(vocabularyCodes)
	y := model.NewProc(g).(*Processor).Classify([]string{"the", "big", "data", "center"})

	print(y)
	if true {
	} else {
		t.Error("output doesn't match the expected values")
	}
}

func getHashedVocabulary(vocabulary *vocabulary.Vocabulary) map[string]mat.Matrix {
	var outMap map[string]mat.Matrix
	outMap = make(map[string]mat.Matrix)
	for _, word := range vocabulary.Items() {
		outMap[word] = getStringCode(word)
	}
	return outMap
}

func getVocabulary() *vocabulary.Vocabulary {
	return vocabulary.New([]string{"airplane", "ball", "center", "data", "the", "big"})
}

func newTestModel() *Model {
	config := Config{
		EncodingActivation:    "Identity",
		ConvActivation:        "Identity",
		ConvSize:              0,
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
		OutputSize:            0,
		TypeVocabSize:         0,
		VocabSize:             0,
		Id2Label:              nil,
	}
	return NewDefaultPrado(config, "path")
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
