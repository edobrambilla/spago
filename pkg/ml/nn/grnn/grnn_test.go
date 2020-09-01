// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package grnn

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

func TestModel_ForwardRecurrent(t *testing.T) {

	model := newTestModel()
	g := ag.NewGraph()
	testGraph := newTestGraph()
	print(testGraph)
	proc := model.NewProc(g)
	proc.(*Processor).SetDirectedGraph(testGraph)
	proc.(*Processor).SetTimeSteps(2)

	// == Forward

	x1 := g.NewVariable(mat.NewVecDense([]float64{0.3, 0.6, -0.1}), true)
	x2 := g.NewVariable(mat.NewVecDense([]float64{0.5, -0.5, 0.9}), true)
	x3 := g.NewVariable(mat.NewVecDense([]float64{0.5, 0.8, -0.3}), true)
	x4 := g.NewVariable(mat.NewVecDense([]float64{0.3, -0.1, -0.4}), true)

	y := proc.Forward(x1, x2, x3, x4)

	if !floats.EqualApprox(y[0].Value().Data(), []float64{
		0.6222744683, 0.2436406261, -0.1713282624,
	}, 1.0e-05) {
		t.Error("The output 0 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[1].Value().Data(), []float64{
		0.7122782943, -0.0761406654, 0.56297994,
	}, 1.0e-05) {
		t.Error("The output 1 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[2].Value().Data(), []float64{
		0.5614723525, 0.239030064, -0.2363710009,
	}, 1.0e-05) {
		t.Error("The output 2 doesn't match the expected values")
	}

	if !floats.EqualApprox(y[3].Value().Data(), []float64{
		0.3268850902, -0.0662579297, 0.0984923482,
	}, 1.0e-05) {
		t.Error("The output 3 doesn't match the expected values")
	}

	// == Backward

	//y[0].PropagateGrad(mat.NewVecDense([]float64{0.3, 0.2, -0.2}))
	//y[1].PropagateGrad(mat.NewVecDense([]float64{0.5, 0.4, 0.0}))
	//y[2].PropagateGrad(mat.NewVecDense([]float64{0.3, -0.7, 0.2}))
	//y[3].PropagateGrad(mat.NewVecDense([]float64{-0.9, -0.7, 0.2}))
	//
	//g.BackwardAll()
	//
	//if !floats.EqualApprox(model.W[0].Grad().Data(), []float64{
	//	0.1622302865, -0.0163271814, 0.0074281477,
	//	0.1060745809, 0.0172205652, 0.0210614703,
	//	-0.1637550022, -0.008649346, -0.0100562922,
	//	-0.0476744545, -0.0950999254, -0.078159356,
	//}, 1.0e-05) {
	//	t.Error("W0 gradients don't match the expected values")
	//}
	//
	//if !floats.EqualApprox(model.W[1].Grad().Data(), []float64{
	//	0.2690607435, -0.0412932047, 0.056239848,
	//	0.1421173997, -0.0805320564, 0.0724676832,
	//	0.10486543, -0.0184519404, 0.0769265757,
	//}, 1.0e-05) {
	//	t.Error("W1 gradients don't match the expected values")
	//}
	//
	//if !floats.EqualApprox(x1.Grad().Data(), []float64{
	//	0.0674194741, 0.0759381594, -0.0527890726, -0.0768017336,
	//}, 1.0e-05) {
	//	t.Error("x1 gradients don't match the expected values")
	//}
	//
	//if !floats.EqualApprox(x2.Grad().Data(), []float64{
	//	0.066802408, 0.0534937184, -0.0731871843, -0.0290116703,
	//}, 1.0e-05) {
	//	t.Error("x2 gradients don't match the expected values")
	//}
	//
	//if !floats.EqualApprox(x3.Grad().Data(), []float64{
	//	0.0299081169, 0.0423251939, -0.0511832432, -0.0088899801,
	//}, 1.0e-05) {
	//	t.Error("x3 gradients don't match the expected values")
	//}
	//
	//if !floats.EqualApprox(x4.Grad().Data(), []float64{
	//	0.0002050635, 0.0268003286, 0.0844189059, -0.1296418032,
	//}, 1.0e-05) {
	//	t.Error("x4 gradients don't match the expected values")
	//}
	//
}

func newTestModel() *Model {
	model := New(3)
	model.WPart.Value().SetData([]float64{
		0.4, 0.2, -0.1,
		0.2, 0.1, 0.3,
		-0.5, 0.2, 0.4,
	})
	model.WPartRec.Value().SetData([]float64{
		0.5, 0.2, 0.1,
		-0.3, -0.2, -0.5,
		0.2, 0.1, 0.3,
	})
	model.WRes.Value().SetData([]float64{
		0.3, 0.9, -0.1,
		-0.1, -0.8, 0.5,
		-0.5, 0.4, 0.9,
	})
	model.WResRec.Value().SetData([]float64{
		-0.7, -0.8, -0.2,
		0.6, 0.8, 0.2,
		0.1, 0.0, 0.7,
	})
	model.WCand.Value().SetData([]float64{
		0.4, 0.2, 0.0,
		0.2, 0.0, 0.7,
		0.1, 0.1, 0.1,
	})
	model.WCandRec.Value().SetData([]float64{
		0.3, -0.8, -0.6,
		0.1, 0.2, -0.7,
		0.5, 0.3, 0.1,
	})
	model.B.Value().SetData([]float64{0.2, 0.1, -0.3})
	return model
}

func newTestGraph() *simple.DirectedMatrix {
	n := 4
	g := make([]graph.Node, n)
	for i := 0; i < n; i++ {
		g[i] = simple.Node(i)
	}

	m := simple.NewDirectedMatrixFrom(g, 0.0, 0.0, 0.0)

	m.SetEdge(simple.Edge{F: g[0], T: g[1]})
	m.SetEdge(simple.Edge{F: g[0], T: g[2]})
	m.SetEdge(simple.Edge{F: g[0], T: g[3]})
	m.SetEdge(simple.Edge{F: g[1], T: g[2]})
	m.SetEdge(simple.Edge{F: g[1], T: g[0]})
	m.SetEdge(simple.Edge{F: g[2], T: g[0]})
	m.SetEdge(simple.Edge{F: g[3], T: g[0]})
	m.SetEdge(simple.Edge{F: g[2], T: g[1]})

	return m
}
