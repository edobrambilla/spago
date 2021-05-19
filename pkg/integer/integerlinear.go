// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package integer

import (
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

// Model contains the serializable parameters.
type IntModel struct {
	W QuantizedInt8Matrix `spago:"type:weights"`
	B QuantizedIntMatrix  `spago:"type:biases"`
}

// New returns a new model with parameters initialized to zeros.
func NewLinearIntModel(lm *linear.Model, q Quantization) *IntModel {
	qb := NewQuantizationClipScaling(q.B, q.Clip, q.scaling*q.scaling)
	model := &IntModel{
		W: q.QuantizeFloatMatrixInt8(lm.W.Value().Rows(), lm.W.Value().Columns(), lm.W.Value().Data()),
		B: qb.QuantizeFloatMatrix(lm.B.Value().Rows(), lm.B.Value().Columns(), lm.B.Value().Data()),
	}
	return model
}

// Forward performs the forward step for each input node and returns the result.
func (m *IntModel) Forward(xs QuantizedInt8Matrix) QuantizedIntMatrix {
	out := MulInt8(m.W, xs)
	stackedB := make([][]int32, len(out.Matrix))
	for i := 0; i < len(stackedB); i++ {
		stackedB[i] = make([]int32, len(out.Matrix[0]))
		for j := 0; j < len(out.Matrix[0]); j++ {
			stackedB[i][j] = m.B.Matrix[i][0]
		}
	}
	quantizedOut := Add(out, QuantizedIntMatrix{
		Matrix:  stackedB,
		Scaling: m.B.Scaling,
	})
	return quantizedOut
}
