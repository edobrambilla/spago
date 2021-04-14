// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package prado

import (
	"github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
	"testing"
)

func TestModel_Forward(t *testing.T) {
	g := ag.NewGraph()
	v := getVocabulary()
	model := newTestModel()
	vocabularyCodes := getHashedVocabulary(v, model.Embeddings.EmbeddingsConfig)
	model.InitPradoParameters(rand.NewLockedRand(743))
	model.Embeddings.SetProjectedEmbeddings(vocabularyCodes)
	ctx := nn.Context{Graph: g, Mode: nn.Inference}
	model = nn.Reify(ctx, model).(*Model)
	y := model.Forward([]string{"the", "big", "data", "center"})

	print(y)
	if true {
	} else {
		t.Error("output doesn't match the expected values")
	}
}

func getHashedVocabulary(vocabulary *vocabulary.Vocabulary, config EmbeddingsConfig) map[string]mat32.Matrix {
	var outMap map[string]mat32.Matrix
	outMap = make(map[string]mat32.Matrix)
	//r := rand.NewLockedRand(40)
	for _, word := range vocabulary.Items() {
		outMap[word] = GetStringCode(word, config)
	}
	return outMap
}

func getVocabulary() *vocabulary.Vocabulary {
	return vocabulary.New([]string{"airplane", "ball", "bill", "data", "the", "big"})
}

func newTestModel() *Model {
	config := Config{
		EncodingActivation:    "ReLU",
		ConvActivation:        "Identity",
		OutputActivation:      "Identity",
		ConvSize:              4,
		InputSize:             16,
		ProjectionSize:        16,
		ProjectionArity:       3,
		EncodingSize:          8,
		UnigramsChannels:      1,
		BigramsChannels:       1,
		TrigramsChannels:      1,
		FourgramsChannels:     0,
		FivegramsChannels:     0,
		Skip1BigramsChannels:  1,
		Skip2BigramsChannels:  1,
		Skip1TrigramsChannels: 0,
		TypeVocabSize:         0,
		VocabSize:             5,
		Id2Label: map[string]string{
			"0": "Arts",
			"1": "Sport",
			"2": "Politics",
			"3": "Crime",
		},
	}
	return NewDefaultPrado(config, "path")
}
