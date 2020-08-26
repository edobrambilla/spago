// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gnn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"gonum.org/v1/gonum/graph/simple"
	"math"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Config struct {
	layers     int
	inputSize  int
	outputSize []int
	Activation ag.OpName
}

type Model struct {
	Config
	W []*nn.Param `type:"weights"`
}

func New(config Config) *Model {
	W := make([]*nn.Param, config.layers)
	W[0] = nn.NewParam(mat.NewEmptyDense(config.inputSize, config.outputSize[0]))
	for i := 1; i < config.layers; i++ {
		W[i] = nn.NewParam(mat.NewEmptyDense(config.outputSize[i-1], config.outputSize[i]))
	}
	return &Model{
		Config: config,
		W:      W,
	}
}

type Processor struct {
	nn.BaseProcessor
	Config
	w               []ag.Node
	AdjacencyMatrix *simple.DirectedMatrix
	// todo whether to enable the concurrent forward computation on the output channel
	// concurrent bool
}

func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	w := make([]ag.Node, len(m.W))

	for i := range m.W {
		w[i] = g.NewWrap(m.W[i])
	}
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
		Config:          m.Config,
		w:               w,
		AdjacencyMatrix: nil,
	}
}

// Return D^-1/2
func (p *Processor) getDegreeMatrix(m *simple.DirectedMatrix) *mat.Dense {
	dim := m.Nodes().Len()
	degreeMatrix := mat.NewEmptyDense(dim, dim)
	for i := 0; i < dim; i++ {
		sum := 0.0 // or 0.0
		for j := 0; j < dim; j++ {
			sum += m.Matrix().At(i, j)
		}
		degreeMatrix.Set(i, i, math.Pow(sum, -0.5))
	}
	return degreeMatrix
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

func (p *Processor) getNormalizationMatrix(A *mat.Dense, D *mat.Dense) *mat.Dense {
	return D.Mul(A).Mul(D).(*mat.Dense)
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	length := len(xs)
	g := p.Graph
	stackedNodes := g.Stack(xs...)
	adjacencyMatrix := p.getAdjacencyMatrix(p.AdjacencyMatrix)
	degreeMatrix := p.getDegreeMatrix(p.AdjacencyMatrix)
	normalizationMatrix := p.getNormalizationMatrix(adjacencyMatrix, degreeMatrix)
	L := g.NewVariable(normalizationMatrix, false)
	H := make([]ag.Node, p.Config.layers)
	H[0] = g.Invoke(p.Activation, g.Mul(L, g.Mul(stackedNodes, p.w[0])))
	for i := 1; i < p.Config.layers; i++ {
		H[i] = g.Invoke(p.Activation, g.Mul(L, g.Mul(H[i-1], p.w[i])))
	}

	out := make([]ag.Node, length)
	for i := 0; i < len(xs); i++ {
		out[i] = g.RowView(H[p.Config.layers-1], i)
	}

	return out
}
