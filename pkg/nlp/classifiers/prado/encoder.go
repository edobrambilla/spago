// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package prado

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Model = &Encoder{}
)

type EncoderConfig struct {
	InputSize   int
	EncodedSize int
	Activation  ag.OpName
}

type Encoder struct {
	nn.BaseModel
	EncoderConfig
	*stack.Model
}

func NewPradoEncoder(config EncoderConfig) *Encoder {
	return &Encoder{
		Model: stack.New(
			linear.New(config.InputSize, config.EncodedSize),
			activation.New(config.Activation),
		),
	}
}

func (p *Encoder) Encode(encoded []ag.Node) []ag.Node {
	return p.Forward(encoded...)
}
