// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package grnn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/gnn"
	"gonum.org/v1/gonum/graph/simple"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &gnn.Processor{}
)

type Model struct {
	WPart    *nn.Param `type:"weights"`
	WPartRec *nn.Param `type:"weights"`
	WRes     *nn.Param `type:"weights"`
	WResRec  *nn.Param `type:"weights"`
	WCand    *nn.Param `type:"weights"`
	WCandRec *nn.Param `type:"weights"`
	B        *nn.Param `type:"biases"`
}

func New(in int) *Model {
	var m Model
	m.WPart = nn.NewParam(mat.NewEmptyDense(in, in))
	m.WPartRec = nn.NewParam(mat.NewEmptyDense(in, in))
	m.WRes = nn.NewParam(mat.NewEmptyDense(in, in))
	m.WResRec = nn.NewParam(mat.NewEmptyDense(in, in))
	m.WCand = nn.NewParam(mat.NewEmptyDense(in, in))
	m.WCandRec = nn.NewParam(mat.NewEmptyDense(in, in))
	m.B = nn.NewParam(mat.NewEmptyVecDense(in))
	return &m
}

type State struct {
	R []ag.Node
	P []ag.Node
	C []ag.Node
	Y []ag.Node
}

type Processor struct {
	nn.BaseProcessor
	wPart           ag.Node
	wPartRec        ag.Node
	wRes            ag.Node
	wResRec         ag.Node
	wCand           ag.Node
	wCandRec        ag.Node
	b               ag.Node
	States          []*State
	AdjacencyMatrix *simple.DirectedMatrix
	Timesteps       int
}

func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
		States:          nil,
		wPart:           g.NewWrap(m.WPart),
		wPartRec:        g.NewWrap(m.WPartRec),
		wRes:            g.NewWrap(m.WRes),
		wResRec:         g.NewWrap(m.WResRec),
		wCand:           g.NewWrap(m.WCand),
		wCandRec:        g.NewWrap(m.WCandRec),
		b:               g.NewWrap(m.B),
		AdjacencyMatrix: nil,
		Timesteps:       1,
	}
}

func (p *Processor) SetTimeSteps(t int) {
	if t < 1 {
		log.Fatal("grnn: TimeSteps must be at least 1")
	}
	p.Timesteps = t
}

func (p *Processor) SetDirectedGraph(matrix *simple.DirectedMatrix) {
	p.AdjacencyMatrix = matrix
}

func (p *Processor) getAdjacencyMatrix(m *simple.DirectedMatrix) *mat.Dense {
	dim := m.Nodes().Len()
	data := make([]float64, dim*dim)
	k := 0
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			data[k] = m.Matrix().At(i, j)
			k++
		}
	}
	adjacencyMatrix := mat.NewDense(dim, dim, data)
	return adjacencyMatrix
}

func (p *Processor) initState(xs []ag.Node) *State {
	length := len(xs)
	s := new(State)
	s.P = make([]ag.Node, length)
	s.R = make([]ag.Node, length)
	s.C = make([]ag.Node, length)
	s.Y = make([]ag.Node, length)
	return s
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	adjacencyMatrix := p.getAdjacencyMatrix(p.AdjacencyMatrix)
	p.States = make([]*State, p.Timesteps)
	p.States[0] = p.forward(xs, adjacencyMatrix)
	for i := 1; i < p.Timesteps; i++ {
		s := p.forward(p.States[i-1].Y, adjacencyMatrix)
		p.States[i] = s
	}
	return p.States[p.Timesteps-1].Y
}

func (p *Processor) forward(x []ag.Node, A *mat.Dense) (s *State) {
	g := p.Graph
	stackedNodes := g.Stack(x...)
	s = p.initState(x)
	a := g.Mul(g.NewVariable(A, false), stackedNodes)
	av := make([]ag.Node, len(x))
	for i := 0; i < len(x); i++ {
		av[i] = g.Add(g.RowView(a, i), p.b)
		s.R[i] = g.Sigmoid(g.Add(g.Mul(av[i], p.wRes), g.Mul(g.T(x[i]), p.wResRec)))
		s.P[i] = g.Sigmoid(g.Add(g.Mul(av[i], p.wPart), g.Mul(g.T(x[i]), p.wPartRec)))
		s.C[i] = g.Tanh(g.Add(g.Mul(av[i], p.wCand), g.Mul(g.T(g.Prod(x[i], s.R[i])), p.wCandRec)))
		s.Y[i] = g.T(g.Add(g.Prod(g.ReverseSub(s.P[i], g.NewScalar(1.0)), x[i]),
			g.Prod(s.P[i], s.C[i])))
	}
	return
}
