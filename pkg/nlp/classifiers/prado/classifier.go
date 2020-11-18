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
	_ nn.Model     = &Classifier{}
	_ nn.Processor = &ClassifierProcessor{}
)

type ClassifierConfig struct {
	TextEncodingSize int
	Labels           []string
	Activation       ag.OpName
}

type Classifier struct {
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

type ClassifierProcessor struct {
	*stack.Processor
}

func (m *Classifier) NewProc(g *ag.Graph) nn.Processor {
	return &ClassifierProcessor{
		Processor: m.Model.NewProc(g).(*stack.Processor),
	}
}

// Predicts return the logits.
func (p *ClassifierProcessor) Predict(xs []ag.Node) []ag.Node {
	return p.Forward(xs...)
}
