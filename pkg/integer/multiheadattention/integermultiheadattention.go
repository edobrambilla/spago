// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package intmultiheadattention

import (
	"github.com/nlpodyssey/spago/pkg/integer"
	"github.com/nlpodyssey/spago/pkg/integer/selfattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention/multiheadattention"
)

// Model contains the serializable parameters.
type Model struct {
	Attention   []*intselfattention.Model
	OutputMerge *integer.IntModel
	NumOfHeads  int // number of heads
	Dm          int // input and output vectors dimension
	Dk          int // hidden vectors dimension (Dm / NumOfHeads)
}

type Output struct {
	Scores  [][][]integer.QuantizedInt
	Context []integer.QuantizedIntMatrix
}

// New returns a new model with parameters initialized to zeros.
func NewFrom(m multiheadattention.Model, startingB int) *Model {
	att := make([]*intselfattention.Model, m.NumOfHeads)
	qo := integer.NewQuantization(startingB, 50)
	for i := 0; i < m.NumOfHeads; i++ {
		att[i] = intselfattention.NewFrom(m.Attention[i], startingB)
	}
	return &Model{
		Attention:   att,
		OutputMerge: integer.NewLinearIntModel(m.OutputMerge, qo),
		NumOfHeads:  m.NumOfHeads,
		Dm:          m.Dm,
		Dk:          m.Dk,
	}
}

func (m *Model) Forward(input integer.QuantizedInt8Matrix) Output {
	var out Output

	heads := make([][]integer.QuantizedIntMatrix, m.NumOfHeads)
	int8heads := make([][]integer.QuantizedInt8Matrix, m.NumOfHeads)
	for h, model := range m.Attention {
		heads[h] = model.Forward(input).Context
		qh := integer.NewQuantizationClipScaling(16, 50, heads[h][0].Scaling)
		for j := 0; j < len(heads[h]); j++ {
			int8heads[h][j] = qh.RequantizeMatrixInt8(heads[h][j])
		}
	}

	concatHeads := make([]integer.QuantizedInt8Matrix, len(input.Matrix))
	for i := 0; i < len(input.Matrix); i++ {
		buf := make([]integer.QuantizedInt8Matrix, len(input.Matrix))
		for j := 0; j < m.NumOfHeads; j++ {
			buf[j] = int8heads[j][i]
		}
		qc := integer.NewQuantizationClipScaling(16, 50, int8heads[0][0].Scaling)
		concatHeads[i] = qc.ConcatInt8(buf...)
	}
	for i := 0; i < len(input.Matrix); i++ {
		out.Context[i] = m.OutputMerge.Forward(concatHeads[i])
	}
	return out
}
