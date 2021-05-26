// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package integer

// Model contains the serializable parameters.
type LayerNormIntModel struct {
	W QuantizedIntMatrix `spago:"type:weights"`
	B QuantizedIntMatrix `spago:"type:biases"`
}

// New returns a new model with parameters initialized to zeros.
func NewLayerNormIntModel(size int, w, b []float32) *LayerNormIntModel {
	q := NewQuantization(16, 50)
	qb := NewQuantizationClipScaling(q.B, q.Clip, q.scaling*q.scaling)
	model := &LayerNormIntModel{
		W: q.QuantizeFloatMatrix(1, size, w),
		B: qb.QuantizeFloatMatrix(1, size, b),
	}
	return model
}

func (m *LayerNormIntModel) Forward(input ...QuantizedIntMatrix) []QuantizedIntMatrix {
	out := make([]QuantizedIntMatrix, len(input))
	for i, x := range input {
		q := NewQuantizationScaling(16, x.Scaling)
		norm := q.IntNormalization(x.Matrix[0])
		qn := NewQuantizationScaling(16, norm[0].Scaling)
		matrix := qn.GetQuantizedMatrix(1, len(x.Matrix[0]), norm)
		prod := Prod(matrix, m.W)
		out[i] = Add(prod, m.B)
	}
	return out
}
