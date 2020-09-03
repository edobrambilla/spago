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

	y[0].PropagateGrad(mat.NewVecDense([]float64{0.3, 0.2, -0.2}))
	y[1].PropagateGrad(mat.NewVecDense([]float64{0.5, 0.4, 0.0}))
	y[2].PropagateGrad(mat.NewVecDense([]float64{0.3, -0.7, 0.2}))
	y[3].PropagateGrad(mat.NewVecDense([]float64{-0.9, -0.7, 0.2}))

	g.BackwardAll()

	if !floats.EqualApprox(model.WCand.Grad().Data(), []float64{
		0.32079091, -0.313665, 0.0922607,
		-0.0129870, -0.0174534, 0.08399622,
		0.26010928, -0.2661428, 0.004631112,
	}, 1.0e-05) {
		t.Error("WCand gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.WCandRec.Grad().Data(), []float64{
		0.09709088, -0.0585832, 0.02841111,
		0.18125055, -0.14477007, 0.02465920,
		0.062012782, 0.30777242, -0.003783331,
	}, 1.0e-05) {
		t.Error("WCandRec gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.WRes.Grad().Data(), []float64{
		0.0299884, -0.0069722, 0.060489119,
		-0.01725852, -0.0029035, 0.060484172,
		0.046097886, -0.0085943, -0.04337469,
	}, 1.0e-05) {
		t.Error("WRes gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.WResRec.Grad().Data(), []float64{
		0.008460725, -0.0060413, 0.034474403,
		0.03728662, -0.00707001, -0.01486535,
		-0.055869141, 0.0018928, 0.03451511,
	}, 1.0e-05) {
		t.Error("WResRec gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.WPart.Grad().Data(), []float64{
		0.0081227, 0.1105733, 0.0202011,
		0.000862, 0.07012200, -0.00002612,
		0.0086104, 0.0247294, 0.0009958337,
	}, 1.0e-05) {
		t.Error("WPart gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.WPartRec.Grad().Data(), []float64{
		0.00419920, 0.0624006, 0.00443294,
		0.00201459, 0.03463037, 0.00438940,
		-0.00059637, 0.004991711, -0.002928727,
	}, 1.0e-05) {
		t.Error("WPartRec gradients don't match the expected values")
	}

	if !floats.EqualApprox(model.B.Grad().Data(), []float64{
		-0.1128803, 0.14028164, 0.01682358,
	}, 1.0e-05) {
		t.Error("B gradients don't match the expected values")
	}

	if !floats.EqualApprox(x1.Grad().Data(), []float64{
		-0.115198506, 0.262924956, -0.03733172,
	}, 1.0e-05) {
		t.Error("x1 gradients don't match the expected values")
	}

	if !floats.EqualApprox(x2.Grad().Data(), []float64{
		0.0208153413, 0.13701687, 0.08982283,
	}, 1.0e-05) {
		t.Error("x2 gradients don't match the expected values")
	}

	if !floats.EqualApprox(x3.Grad().Data(), []float64{
		0.3617735698, -0.22323505, 0.19314677,
	}, 1.0e-05) {
		t.Error("x3 gradients don't match the expected values")
	}

	if !floats.EqualApprox(x4.Grad().Data(), []float64{
		0.0125651907, -0.244330273, -0.05244557,
	}, 1.0e-05) {
		t.Error("x4 gradients don't match the expected values")
	}

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
