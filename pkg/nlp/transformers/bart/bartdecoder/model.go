// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bartdecoder

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartconfig"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/posembeddings"
	"github.com/nlpodyssey/spago/pkg/utils"
)

var (
	_ nn.Module = &Model{}
)

// Model implements a BART decoder.
type Model struct {
	nn.BaseModel
	Config                      bartconfig.Config
	LearnedPositionalEmbeddings *posembeddings.LearnedPositionalEmbeddings
	Layers                      *stack.Model
	EmbeddingLayerNorm          *layernorm.Model
	LayerNorm                   *layernorm.Model
}

// New returns a new BART decoder Model.
func New(config bartconfig.Config) *Model {
	if config.StaticPositionEmbeddings {
		panic("bart: static position embeddings not implemented.")
	}
	if config.ScaleEmbedding {
		panic("bart: scale embedding not implemented.")
	}
	learnedPositionalEmbeddings := make([]nn.Param, config.MaxPositionEmbeddings+config.ExtraPosEmbedding)
	for i := 0; i < len(learnedPositionalEmbeddings); i++ {
		learnedPositionalEmbeddings[i] = nn.NewParam(mat.NewEmptyVecDense(config.DModel))
	}
	return &Model{
		BaseModel: nn.BaseModel{FullSeqProcessing: true},
		Config:    config,
		LearnedPositionalEmbeddings: posembeddings.NewLearnedPositionalEmbeddings(
			posembeddings.Config{
				NumEmbeddings: config.VocabSize,
				EmbeddingDim:  config.DModel,
				PaddingIDX:    0, // TODO
				Offset:        config.ExtraPosEmbedding,
			}),
		EmbeddingLayerNorm: layernorm.New(config.DModel),
		Layers: stack.Make(config.DecoderLayers, func(_ int) nn.Module {
			return NewLayer(config)
			// add LayerDrop to skip layers during training (see https://arxiv.org/abs/1909.11556 for description)
		}),
		LayerNorm: layernorm.New(config.DModel),
	}
}

// Decode performs the forward step for each input and returns the result.
func (m *Model) Decode(xs, encoded []ag.Node) []ag.Node {
	embedPos := m.LearnedPositionalEmbeddings.Encode(utils.MakeIndices(len(xs)))
	ys := m.add(xs, embedPos)
	ys = m.EmbeddingLayerNorm.Forward(ys...)
	// ys = m.Dropout(ys)

	for _, layer := range m.Layers.Layers {
		ys = layer.(*Layer).Process(ys, encoded)
		// TODO: save all hidden states into the processor to allow a later access
	}

	if m.Config.FinalLayerNorm {
		ys = m.LayerNorm.Forward(ys...)
	}
	return ys
}

func (m *Model) add(a []ag.Node, b []ag.Node) []ag.Node {
	c := make([]ag.Node, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = m.Graph().Add(a[i], b[i])
	}
	return c
}

// Forward is not implemented for BART decoder Processor (it always panics).
// You should use Decode instead.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	panic("bartdecoder: Forward() not implemented; use Decode() instead.")
}
