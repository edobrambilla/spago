// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package intmultiheadattention

import (
	"github.com/nlpodyssey/spago/pkg/integer"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention/multiheadattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention/selfattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_LinearMultiheadAttention(t *testing.T) {
	b := 8
	attModels := make([]*selfattention.Model, 2)
	attModels[0] = newTestModelSelfAttention()
	attModels[1] = newTestModelSelfAttention2()
	m := multiheadattention.Model{
		Attention:   attModels,
		OutputMerge: newTestModel(),
		NumOfHeads:  2,
		Dm:          6,
		Dk:          3,
	}
	q := integer.NewQuantization(b, 50)
	model := NewFrom(m, b)
	x1 := []float32{-0.8, -0.9, -0.9, 1.0}
	x2 := []float32{0.8, -0.3, 0.5, 0.3}
	x3 := []float32{-0.2, 0.7, 0.2, 0.4}
	xs := append(x1, x2...)
	xs = append(xs, x3...)
	qin := q.QuantizeFloatMatrixInt8(3, 4, xs)
	transposedqin := integer.TransposeInt8(qin)
	output := model.Forward(transposedqin)
	assert.Equal(t, output.Context[0].Matrix[0][0], int32(104))
}

func newTestModel() *linear.Model {
	model := linear.New(6, 3)
	model.W.Value().SetData([]mat.Float{
		0.5, 0.6, -0.8, -0.6, 0.3, 0.0,
		0.7, -0.4, 0.1, -0.8, -0.3, -0.8,
		0.7, -0.7, 0.3, 0.5, 0.1, 0.1,
	})
	model.B.Value().SetData([]mat.Float{0.4, 0.0, -0.3})
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

func newTestModelSelfAttention2() *selfattention.Model {
	model := selfattention.New(selfattention.Config{
		InputSize:   4,
		QuerySize:   3,
		KeySize:     3,
		ValueSize:   3,
		ScaleFactor: 3.0,
	})
	model.Value.W.Value().SetData([]mat.Float{
		-0.8, 0.6, 0.8, 0.3,
		-0.4, -0.1, 0.7, -0.2,
		-0.3, 0.4, 0.0, 0.0,
	})
	model.Value.B.Value().SetData([]mat.Float{-0.4, 0.9, -0.7})
	model.Key.W.Value().SetData([]mat.Float{
		0.6, -0.2, 0.1, -0.9,
		0.1, -0.1, 0.3, -0.8,
		0.8, 0.5, 0.1, 0.8,
	})
	model.Key.B.Value().SetData([]mat.Float{-0.3, -0.1, -0.8})
	model.Query.W.Value().SetData([]mat.Float{
		-0.2, 0.0, -0.9, 0.6,
		0.9, -0.9, 0.9, 0.0,
		-0.7, 0.5, 0.1, -0.1,
	})
	model.Query.B.Value().SetData([]mat.Float{-0.3, 0.6, -0.9})
	return model
}
