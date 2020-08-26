// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gnn

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/graph"
	//"github.com/nlpodyssey/spago/pkg/ml/ag"
	//"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/graph/simple"
	"testing"
)

func TestModel_Forward(t *testing.T) {

	model := newTestModel()
	g := ag.NewGraph()
	testGraph := newTestGraph()
	proc := model.NewProc(g)
	proc.(*Processor).SetDirectedGraph(testGraph)

	// == Forward

	x1 := g.NewVariable(mat.NewVecDense([]float64{0.1, 0.3, -0.1, -0.1}), true)
	x2 := g.NewVariable(mat.NewVecDense([]float64{0.5, 0.2, -0.9, 0.2}), true)
	x3 := g.NewVariable(mat.NewVecDense([]float64{0.6, 0.2, 0.2, -0.5}), true)
	x4 := g.NewVariable(mat.NewVecDense([]float64{-0.6, -0.7, -0.8, -0.9}), true)
	x5 := g.NewVariable(mat.NewVecDense([]float64{0.1, 0.3, 0.5, -0.1}), true)
	x6 := g.NewVariable(mat.NewVecDense([]float64{0.4, 0.4, 0.1, 0.1}), true)

	y := proc.Forward(x1, x2, x3, x4, x5, x6)

	if !floats.EqualApprox(y[0].Value().Data(), []float64{
		0.3497545891, 0.1256949019, 0,
	}, 1.0e-05) {
		t.Error("The output 0 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[1].Value().Data(), []float64{
		0.2587385912, 0.0430940386, 0,
	}, 1.0e-05) {
		t.Error("The output 1 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[2].Value().Data(), []float64{
		0.2810648628, 0.1178418831, 0,
	}, 1.0e-05) {
		t.Error("The output 2 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[3].Value().Data(), []float64{
		0.2303849764, 0.1168909212, 0.0017021841,
	}, 1.0e-05) {
		t.Error("The output 3 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[4].Value().Data(), []float64{
		0.1714193948, 0.1143694886, 0.0638586488,
	}, 1.0e-05) {
		t.Error("The output 4 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[5].Value().Data(), []float64{
		0.1714193948, 0.1143694886, 0.0638586488,
	}, 1.0e-05) {
		t.Error("The output 5 doesn't match the expected values")
	}

	// == Backward

	y[0].PropagateGrad(mat.NewVecDense([]float64{0.3, 0.2, -0.2}))
	y[1].PropagateGrad(mat.NewVecDense([]float64{0.5, 0.4, 0.0}))
	y[2].PropagateGrad(mat.NewVecDense([]float64{0.3, -0.7, 0.2}))
	y[3].PropagateGrad(mat.NewVecDense([]float64{-0.9, -0.7, 0.2}))
	y[4].PropagateGrad(mat.NewVecDense([]float64{0.7, 0.2, 0.1}))
	y[5].PropagateGrad(mat.NewVecDense([]float64{0.4, 0.5, 0.2}))

	g.BackwardAll()

	if !floats.EqualApprox(model.W[0].Grad().Data(), []float64{
		0.1622302865, -0.0163271814, 0.0074281477,
		0.1060745809, 0.0172205652, 0.0210614703,
		-0.1637550022, -0.008649346, -0.0100562922,
		-0.0476744545, -0.0950999254, -0.078159356,
	}, 1.0e-05) {
		t.Error("W0 gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.W[1].Grad().Data(), []float64{
		0.2690607435, -0.0412932047, 0.056239848,
		0.1421173997, -0.0805320564, 0.0724676832,
		0.10486543, -0.0184519404, 0.0769265757,
	}, 1.0e-05) {
		t.Error("W1 gradients don't match the expected values")
	}

	if !floats.EqualApprox(x1.Grad().Data(), []float64{
		0.0674194741, 0.0759381594, -0.0527890726, -0.0768017336,
	}, 1.0e-05) {
		t.Error("x1 gradients don't match the expected values")
	}

	if !floats.EqualApprox(x2.Grad().Data(), []float64{
		0.066802408, 0.0534937184, -0.0731871843, -0.0290116703,
	}, 1.0e-05) {
		t.Error("x2 gradients don't match the expected values")
	}

	if !floats.EqualApprox(x3.Grad().Data(), []float64{
		0.0299081169, 0.0423251939, -0.0511832432, -0.0088899801,
	}, 1.0e-05) {
		t.Error("x3 gradients don't match the expected values")
	}

	if !floats.EqualApprox(x4.Grad().Data(), []float64{
		0.0002050635, 0.0268003286, 0.0844189059, -0.1296418032,
	}, 1.0e-05) {
		t.Error("x4 gradients don't match the expected values")
	}

	if !floats.EqualApprox(x5.Grad().Data(), []float64{
		-0.0033701943, 0.025471633, 0.1062986853, -0.1514780087,
	}, 1.0e-05) {
		t.Error("x5 gradients don't match the expected values")
	}

	if !floats.EqualApprox(x6.Grad().Data(), []float64{
		-0.0033701943, 0.025471633, 0.1062986853, -0.1514780087,
	}, 1.0e-05) {
		t.Error("x6 gradients don't match the expected values")
	}
}

func newTestModel() *Model {
	model := New(Config{
		layers:     2,
		inputSize:  4,
		outputSize: []int{3, 3},
		Activation: ag.OpReLU,
	})
	model.W[0].Value().SetData([]float64{
		0.4, 0.3, -0.5,
		0.3, -0.2, 0.6,
		-0.4, 0.5, 0.4,
		-0.2, -0.7, -0.6,
	})

	model.W[1].Value().SetData([]float64{
		0.8, 0.1, -0.7,
		0.3, 0.4, 0.6,
		0.4, 0.3, 0.2,
	})
	return model
}

func newTestGraph() *simple.DirectedMatrix {
	n := 6
	g := make([]graph.Node, n)
	for i := 0; i < n; i++ {
		g[i] = simple.Node(i)
	}

	m := simple.NewDirectedMatrixFrom(g, 0.0, 1.0, 0.0)

	m.SetEdge(simple.Edge{F: g[0], T: g[1]})
	m.SetEdge(simple.Edge{F: g[0], T: g[2]})
	m.SetEdge(simple.Edge{F: g[0], T: g[3]})
	m.SetEdge(simple.Edge{F: g[3], T: g[4]})
	m.SetEdge(simple.Edge{F: g[3], T: g[5]})
	m.SetEdge(simple.Edge{F: g[4], T: g[5]})
	m.SetEdge(simple.Edge{F: g[1], T: g[0]})
	m.SetEdge(simple.Edge{F: g[2], T: g[0]})
	m.SetEdge(simple.Edge{F: g[3], T: g[0]})
	m.SetEdge(simple.Edge{F: g[4], T: g[3]})
	m.SetEdge(simple.Edge{F: g[5], T: g[3]})
	m.SetEdge(simple.Edge{F: g[5], T: g[4]})

	return m
}
