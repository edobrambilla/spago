// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package integer

import (
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

// Model contains the serializable parameters.
type IntModel struct {
	W QuantizedIntMatrix `spago:"type:weights"`
	B QuantizedIntMatrix `spago:"type:biases"`
}

// New returns a new model with parameters initialized to zeros.
func NewLinearIntModel(lm *linear.Model, q Quantization) *IntModel {
	qb := NewQuantizationClipScaling(q.b, q.clip, q.scaling*q.scaling)
	model := &IntModel{
		W: q.QuantizeFloatMatrix(lm.W.Value().Rows(), lm.W.Value().Columns(), lm.W.Value().Data()),
		B: qb.QuantizeFloatMatrix(lm.B.Value().Rows(), lm.B.Value().Columns(), lm.B.Value().Data()),
	}
	return model
}

// Forward performs the forward step for each input node and returns the result.
func (m *IntModel) Forward(xs QuantizedIntMatrix) QuantizedIntMatrix {
	out := Mul(m.W, xs)
	stackedB := make([][]int, len(out.matrix))
	for i := 0; i < len(stackedB); i++ {
		stackedB[i] = make([]int, len(out.matrix[0]))
		for j := 0; j < len(out.matrix[0]); j++ {
			stackedB[i][j] = m.B.matrix[i][0]
		}
	}
	quantizedOut := Add(out, QuantizedIntMatrix{
		matrix:  stackedB,
		scaling: m.B.scaling,
	})
	return quantizedOut
}
