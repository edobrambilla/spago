// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package prado

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model     = &TextEncoder{}
	_ nn.Processor = &TextEncoderProcessor{}
)

type TextEncoder struct{}

func NewPradoTextEncoder() *TextEncoder {
	return &TextEncoder{}
}

type TextEncoderProcessor struct {
	nn.BaseProcessor
}

func (m *TextEncoder) NewProc(ctx nn.Context) nn.Processor {
	return &TextEncoderProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
	}
}

func (p *TextEncoderProcessor) Encode(projectedFeatures [][]ag.Node, attentionFeatures [][]ag.Node) ag.Node {
	e := make([]ag.Node, len(projectedFeatures))
	for channel, projectedFeature := range projectedFeatures {
		en := projectedFeature[0]
		for i := 1; i < len(projectedFeature); i++ {
			en = p.Graph.Add(en, p.Graph.Prod(projectedFeature[i], attentionFeatures[channel][i]))
		}
		e[channel] = en
	}
	return p.Graph.Concat(e...)
}

func (p *TextEncoderProcessor) Forward(_ ...ag.Node) []ag.Node {
	panic("Prado: Forward() method not implemented. Use Encode() instead.")
}
