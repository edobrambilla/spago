// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package prado

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model = &TextEncoder{}
)

type TextEncoder struct {
	nn.BaseModel
}

func NewPradoTextEncoder() *TextEncoder {
	return &TextEncoder{}
}

func (p *TextEncoder) Encode(projectedFeatures [][]ag.Node, attentionFeatures [][]ag.Node) ag.Node {
	e := make([]ag.Node, len(projectedFeatures))
	for channel, projectedFeature := range projectedFeatures {
		en := p.Graph().Prod(projectedFeature[0], attentionFeatures[channel][0])
		for i := 1; i < len(projectedFeature); i++ {
			en = p.Graph().Add(en, p.Graph().Prod(projectedFeature[i], attentionFeatures[channel][i]))
		}
		e[channel] = en
	}
	return p.Graph().Concat(e...)
}
