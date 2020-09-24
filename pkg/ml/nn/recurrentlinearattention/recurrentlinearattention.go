// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package recurrentlinearattention

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Config struct {
	InputSize int
}

type Model struct {
	Config
	Wk *nn.Param `type:"weights"`
	Bk *nn.Param `type:"biases"`
	Wv *nn.Param `type:"weights"`
	Bv *nn.Param `type:"biases"`
	Wq *nn.Param `type:"weights"`
	Bq *nn.Param `type:"biases"`
}

func New(config Config) *Model {
	var m Model
	m.Wk = nn.NewParam(mat.NewEmptyDense(config.InputSize, config.InputSize))
	m.Bk = nn.NewParam(mat.NewEmptyVecDense(config.InputSize))
	m.Wv = nn.NewParam(mat.NewEmptyDense(config.InputSize, config.InputSize))
	m.Bv = nn.NewParam(mat.NewEmptyVecDense(config.InputSize))
	m.Wq = nn.NewParam(mat.NewEmptyDense(config.InputSize, config.InputSize))
	m.Bq = nn.NewParam(mat.NewEmptyVecDense(config.InputSize))
	return &m
}

type State struct {
	S ag.Node
	Z ag.Node
	Y ag.Node
}

type Processor struct {
	nn.BaseProcessor
	wK     ag.Node
	bK     ag.Node
	wV     ag.Node
	bV     ag.Node
	wQ     ag.Node
	bQ     ag.Node
	eps    ag.Node
	States []*State
}

func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
		States: nil,
		wK:     g.NewWrap(m.Wk),
		bK:     g.NewWrap(m.Bk),
		wV:     g.NewWrap(m.Wv),
		bV:     g.NewWrap(m.Bv),
		wQ:     g.NewWrap(m.Wq),
		bQ:     g.NewWrap(m.Bq),
		eps:    g.NewScalar(0.00000000000001),
	}
}

func (p *Processor) SetInitialState(state *State) {
	if len(p.States) > 0 {
		log.Fatal("lstm: the initial state must be set before any input")
	}
	p.States = append(p.States, state)
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		s := p.forward(x)
		p.States = append(p.States, s)
		ys[i] = s.Y
	}
	return ys
}

func (p *Processor) LastState() *State {
	n := len(p.States)
	if n == 0 {
		return nil
	}
	return p.States[n-1]
}

func (p *Processor) forward(x ag.Node) (s *State) {
	g := p.Graph
	s = new(State)
	sPrev, zPrev := p.prev()
	key := nn.Affine(g, p.bK, p.wK, x)
	value := nn.Affine(g, p.bV, p.wV, x)
	query := nn.Affine(g, p.bQ, p.wQ, x)
	one := g.NewScalar(1.0)
	akey := g.AddScalar(g.ELU(key, g.NewScalar(1.0)), one)
	aquery := g.AddScalar(g.ELU(query, g.NewScalar(1.0)), one)
	if sPrev != nil {
		s.S = g.Add(sPrev, g.Mul(akey, g.T(value)))
	} else {
		s.S = g.Mul(akey, g.T(value))
	}
	if zPrev != nil {
		s.Z = g.Add(zPrev, akey)
	} else {
		s.Z = akey
	}
	n := g.Mul(g.T(aquery), s.S)
	d := g.AddScalar(g.Dot(aquery, s.Z), p.eps)
	v := g.DivScalar(n, d)
	s.Y = g.Add(x, v)
	return
}

func (p *Processor) prev() (sPrev, zPrev ag.Node) {
	s := p.LastState()
	if s != nil {
		sPrev = s.S
		zPrev = s.Z
	}
	return
}
