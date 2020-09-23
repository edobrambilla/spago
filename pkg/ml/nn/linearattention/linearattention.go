// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package linearattention

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Self-Attention
type Model struct {
	Config
	Query *linear.Model
	Key   *linear.Model
	Value *linear.Model
}

type Config struct {
	InputSize int
	QuerySize int
	KeySize   int
	ValueSize int
}

func New(config Config) *Model {
	return &Model{
		Config: config,
		Query:  linear.New(config.InputSize, config.QuerySize),
		Key:    linear.New(config.InputSize, config.KeySize),
		Value:  linear.New(config.InputSize, config.ValueSize),
	}
}

type Processor struct {
	nn.BaseProcessor
	scaleFactor float64
	query       *linear.Processor
	key         *linear.Processor
	value       *linear.Processor
}

func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		query: m.Query.NewProc(g).(*linear.Processor),
		key:   m.Key.NewProc(g).(*linear.Processor),
		value: m.Value.NewProc(g).(*linear.Processor),
	}
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	length := len(xs)
	g := p.Graph
	qs := p.query.Forward(xs...)
	ks := p.key.Forward(xs...)
	vs := p.value.Forward(xs...)
	context := make([]ag.Node, length)
	aks := make([]ag.Node, length)
	aqs := make([]ag.Node, length)
	ksum := g.NewVariable(mat.NewEmptyVecDense(ks[0].Value().Size()), true)
	for i := range ks {
		aks[i] = g.AddScalar(g.ELU(ks[i], g.NewScalar(1.0)), g.NewScalar(1.0))
		aqs[i] = g.AddScalar(g.ELU(qs[i], g.NewScalar(1.0)), g.NewScalar(1.0))
		ksum = g.Add(ksum, aks[i])
	}

	keys := g.T(g.Stack(aks...))
	values := g.Stack(vs...)

	kv := g.Mul(keys, values)

	for i, q := range aqs {
		context[i] = g.DivScalar(g.Mul(g.T(q), kv), g.Dot(aqs[i], ksum))
	}
	return context
}
