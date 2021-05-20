// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package intselfattention

import (
	"github.com/nlpodyssey/spago/pkg/integer"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention/selfattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_LinearForward(t *testing.T) {
	q := integer.NewQuantization(8, 50)
	model := newTestModel()
	intModel := integer.NewLinearIntModel(model, integer.NewQuantization(8, 50))
	x1 := []float32{-0.8, -0.9, -0.9}
	x2 := []float32{0.8, -0.3, 0.5}
	x3 := []float32{-0.2, 0.7, 0.2}
	xs := append(x1, x2...)
	xs = append(xs, x3...)
	c := intModel.Forward(q.QuantizeFloatMatrixInt8(3, 3, xs))
	assert.Equal(t, c.Matrix[0], []int32{14, -27, 0})
	assert.Equal(t, c.Matrix[1], []int32{30, -1, 25})
	assert.Equal(t, c.Matrix[2], []int32{-32, 11, -21})
	assert.Equal(t, c.Matrix[3], []int32{16, 23, 24})
}

func Test_LinearSelfAttention(t *testing.T) {
	b := 8
	q := integer.NewQuantization(b, 50)
	model := NewFrom(newTestModelSelfAttention(), b)
	x1 := []float32{-0.8, -0.9, -0.9, 1.0}
	x2 := []float32{0.8, -0.3, 0.5, 0.3}
	x3 := []float32{-0.2, 0.7, 0.2, 0.4}
	xs := append(x1, x2...)
	xs = append(xs, x3...)
	qin := q.QuantizeFloatMatrixInt8(3, 4, xs)
	transposedqin := integer.TransposeInt8(qin)
	output := model.Forward(transposedqin)
	assert.Equal(t, output.Context[0].Matrix[0][0], int32(104))
	assert.Equal(t, output.Context[0].Matrix[1][0], int32(-115))
	assert.Equal(t, output.Context[0].Matrix[2][0], int32(-55))
	assert.Equal(t, output.Context[1].Matrix[0][0], int32(104))
	assert.Equal(t, output.Context[1].Matrix[1][0], int32(-115))
	assert.Equal(t, output.Context[1].Matrix[2][0], int32(-55))
	assert.Equal(t, output.Context[2].Matrix[0][0], int32(104))
	assert.Equal(t, output.Context[2].Matrix[1][0], int32(-76))
	assert.Equal(t, output.Context[2].Matrix[2][0], int32(-43))
	assert.Equal(t, output.Scores[0][0].Value, int32(11468))
	assert.Equal(t, output.Scores[0][1].Value, int32(4029))
	assert.Equal(t, output.Scores[0][2].Value, int32(3230))
	assert.Equal(t, output.Scores[1][0].Value, int32(9800))
	assert.Equal(t, output.Scores[1][1].Value, int32(4142))
	assert.Equal(t, output.Scores[1][2].Value, int32(4785))
	assert.Equal(t, output.Scores[2][0].Value, int32(7897))
	assert.Equal(t, output.Scores[2][1].Value, int32(5281))
	assert.Equal(t, output.Scores[2][2].Value, int32(5549))
	assert.InDelta(t, output.Context[0].Scaling, 0.007538794, 1.0e-6)
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
		ScaleFactor: 3.0,
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
