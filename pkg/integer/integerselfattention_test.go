// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package integer

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention/selfattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_LinearForward(t *testing.T) {
	q := NewQuantization(12, 50)
	g := ag.NewGraph()
	model := newTestModel()
	intModel := NewLinearIntModel(model, NewQuantization(12, 50))
	xs := make([]ag.Node, 3)
	xs[0] = g.NewVariable(mat.NewVecDense([]mat.Float{-0.8, -0.9, -0.9}), false)
	xs[1] = g.NewVariable(mat.NewVecDense([]mat.Float{0.8, -0.3, 0.5}), false)
	xs[2] = g.NewVariable(mat.NewVecDense([]mat.Float{-0.2, 0.7, 0.2}), false)
	stackedIn := Stack(g, q, xs)
	c := intModel.Forward(stackedIn)
	assert.Equal(t, c.matrix[0], []int{1235, 1458, 3764})
	assert.Equal(t, c.matrix[1], []int{1458, -6012, 3505})
	assert.Equal(t, c.matrix[2], []int{-1874, 2503, -4990})
	assert.Equal(t, c.matrix[3], []int{-2464, -4048, 1651})
}

func Test_LinearSelfAttention(t *testing.T) {
	b := 8
	q := NewQuantization(b, 50)
	g := ag.NewGraph()
	xs := make([]ag.Node, 3)
	model := NewFrom(newTestModelSelfAttention(), b)
	xs[0] = g.NewVariable(mat.NewVecDense([]mat.Float{-0.8, -0.9, -0.9, 1.0}), false)
	xs[1] = g.NewVariable(mat.NewVecDense([]mat.Float{0.8, -0.3, 0.5, 0.3}), false)
	xs[2] = g.NewVariable(mat.NewVecDense([]mat.Float{-0.2, 0.7, 0.2, 0.4}), false)
	stackedIn := Stack(g, q, xs)
	output := model.Forward(stackedIn)
	print(output.context[0].matrix) // todo check results using int8/int32
}

func newTestModel() *linear.Model {
	model := linear.New(3, 4)
	model.W.Value().SetData([]mat.Float{
		0.5, 0.6, -0.8,
		-0.6, 0.7, -0.4,
		0.1, -0.8, 0.7,
		-0.7, 0.3, 0.5,
	})
	model.B.Value().SetData([]mat.Float{0.4, 0.0, -0.3, -0.2})
	return model
}

func newTestModelSelfAttention() *selfattention.Model {
	model := selfattention.New(selfattention.Config{
		InputSize:   4,
		QuerySize:   3,
		KeySize:     3,
		ValueSize:   3,
		ScaleFactor: 1.0 / mat.Sqrt(3.0),
	})
	model.Value.W.Value().SetData([]mat.Float{
		0.5, 0.6, -0.8, 0.7,
		-0.4, 0.1, 0.7, -0.7,
		0.3, 0.8, -0.9, 0.0,
	})
	model.Value.B.Value().SetData([]mat.Float{0.4, 0.0, -0.3})
	model.Key.W.Value().SetData([]mat.Float{
		0.7, -0.2, -0.1, 0.2,
		-0.1, -0.1, 0.3, -0.2,
		0.6, 0.1, 0.9, 0.3,
	})
	model.Key.B.Value().SetData([]mat.Float{0.8, -0.2, -0.5})
	model.Query.W.Value().SetData([]mat.Float{
		-0.8, -0.6, 0.2, 0.5,
		0.7, -0.6, -0.3, 0.6,
		-0.3, 0.3, 0.4, -0.8,
	})
	model.Query.B.Value().SetData([]mat.Float{0.3, 0.5, -0.7})
	return model
}
