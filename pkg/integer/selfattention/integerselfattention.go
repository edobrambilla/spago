// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package selfattention

import (
	"github.com/nlpodyssey/spago/pkg/integer"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention/selfattention"
)

// Model contains the serializable parameters.
type Model struct {
	Query             *integer.IntModel
	Key               *integer.IntModel
	Value             *integer.IntModel
	QueryQuantization *integer.Quantization
	KeyQuantization   *integer.Quantization
	ValueQuantization *integer.Quantization
	ScaleFactor       mat.Float
}

//// Config provides configuration settings for a Self-Attention Model.
//type Config struct {
//	InputSize     int
//	QuerySize     int
//	KeySize       int
//	ValueSize     int
//	ScaleFactor   mat.Float
//}

type Output struct {
	scores  [][]integer.QuantizedInt
	context []integer.QuantizedIntMatrix
}

// New returns a new model with parameters initialized to zeros.
func NewFrom(m *selfattention.Model, startingB int) *Model {
	qq := integer.NewQuantization(startingB, 50)
	qk := integer.NewQuantization(startingB, 50)
	qv := integer.NewQuantization(startingB, 50)
	return &Model{
		Query:             integer.NewLinearIntModel(m.Query, qq),
		Key:               integer.NewLinearIntModel(m.Key, qk),
		Value:             integer.NewLinearIntModel(m.Value, qv),
		QueryQuantization: &qq,
		KeyQuantization:   &qk,
		ValueQuantization: &qv,
		ScaleFactor:       1.0 / mat.Sqrt(3.0),
	}
}

func (m *Model) Forward(input integer.QuantizedInt8Matrix) Output {
	out := Output{
		scores:  make([][]integer.QuantizedInt, len(input.Matrix[0])),
		context: make([]integer.QuantizedIntMatrix, len(input.Matrix[0])),
	}
	qkv := m.GetQKV(input)
	// to int 8
	qkvQuantization := integer.NewQuantizationScaling(m.ValueQuantization.B, qkv.Values.Scaling)
	i8Queries := qkvQuantization.RequantizeMatrixInt8(qkv.Queries)
	i8Keys := qkvQuantization.RequantizeMatrixInt8(qkv.Keys)
	i8Values := qkvQuantization.RequantizeMatrixInt8(qkv.Values)

	tQueries := integer.TransposeInt8(i8Queries)
	scores := integer.MulInt8(tQueries, i8Keys)
	qkquantization := integer.NewQuantizationScaling(m.QueryQuantization.B, scores.Scaling)
	scaledScores := integer.ProdScalar(scores, qkquantization.Quantize(m.ScaleFactor))
	scoresquantization := integer.NewQuantizationClipScaling(m.QueryQuantization.B, m.QueryQuantization.Clip, scaledScores.Scaling)
	rescaledScores := scoresquantization.RequantizeMatrix(scaledScores, 12)
	rescaledscoresquantization := integer.NewQuantizationClipScaling(12, 50, rescaledScores.Scaling)
	for i, attscores := range rescaledScores.Matrix {
		out.scores[i] = rescaledscoresquantization.IntSoftmax(attscores)
		softmaxquantization := integer.NewQuantizationClipScaling(16, m.QueryQuantization.Clip,
			out.scores[i][0].Scaling)
		scoresm := softmaxquantization.GetQuantizedMatrix(len(out.scores[i]), 1, out.scores[i])
		//to int 8
		i8Scores := softmaxquantization.RequantizeMatrixInt8(scoresm)
		out.context[i] = integer.MulInt8(i8Values, i8Scores)
	}
	return out
}

func (m *Model) GetQKV(input integer.QuantizedInt8Matrix) integer.IntQKV {
	return integer.IntQKV{
		Queries: m.Query.Forward(input),
		Keys:    m.Key.Forward(input),
		Values:  m.Value.Forward(input),
	}
}
