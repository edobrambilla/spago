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
	_ nn.Model = &Classifier{}
)

type ClassifierConfig struct {
	TextEncodingSize int
	Labels           []string
	Activation       ag.OpName
}

type Classifier struct {
	nn.BaseModel
	config ClassifierConfig
	*stack.Model
}

func NewPradoClassifier(config ClassifierConfig) *Classifier {
	return &Classifier{
		config: config,
		Model: stack.New(
			linear.New(config.TextEncodingSize, len(config.Labels)),
			activation.New(config.Activation),
		),
	}
}

// Predicts return the logits.
func (p *Classifier) Predict(xs []ag.Node) []ag.Node {
	return p.Forward(xs...)
}