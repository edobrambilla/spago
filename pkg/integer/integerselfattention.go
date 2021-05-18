// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package integer

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention/selfattention"
)

// Model contains the serializable parameters.
type Model struct {
	Query             *IntModel
	Key               *IntModel
	Value             *IntModel
	QueryQuantization *Quantization
	KeyQuantization   *Quantization
	ValueQuantization *Quantization
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
	scores  [][]QuantizedInt
	context []QuantizedIntMatrix
}

// New returns a new model with parameters initialized to zeros.
func NewFrom(m *selfattention.Model, startingB int) *Model {
	qq := NewQuantization(startingB, 50)
	qk := NewQuantization(startingB, 50)
	qv := NewQuantization(startingB, 50)
	return &Model{
		Query:             NewLinearIntModel(m.Query, qq),
		Key:               NewLinearIntModel(m.Key, qk),
		Value:             NewLinearIntModel(m.Value, qv),
		QueryQuantization: &qq,
		KeyQuantization:   &qk,
		ValueQuantization: &qv,
		ScaleFactor:       1.0 / mat.Sqrt(3.0),
	}
}

func (m *Model) Forward(input QuantizedInt8Matrix) Output {
	out := Output{
		scores:  make([][]QuantizedInt, len(input.matrix[0])),
		context: make([]QuantizedIntMatrix, len(input.matrix[0])),
	}
	qkv := m.GetQKV(input)
	// to int 8
	qkvQuantization := NewQuantizationScaling(m.ValueQuantization.b, qkv.Values.scaling)
	i8Queries := qkvQuantization.RequantizeMatrixInt8(qkv.Queries)
	i8Keys := qkvQuantization.RequantizeMatrixInt8(qkv.Keys)
	i8Values := qkvQuantization.RequantizeMatrixInt8(qkv.Values)

	tQueries := TransposeInt8(i8Queries)
	scores := MulInt8(tQueries, i8Keys)
	qkquantization := NewQuantizationScaling(m.QueryQuantization.b, scores.scaling)
	scaledScores := ProdScalar(scores, qkquantization.Quantize(m.ScaleFactor))
	scoresquantization := NewQuantizationClipScaling(m.QueryQuantization.b, m.QueryQuantization.clip, scaledScores.scaling)
	rescaledScores := scoresquantization.RequantizeMatrix(scaledScores, 12)
	rescaledscoresquantization := NewQuantizationClipScaling(12, 50, rescaledScores.scaling)
	for i, attscores := range rescaledScores.matrix {
		out.scores[i] = rescaledscoresquantization.IntSoftmax(attscores)
		softmaxquantization := NewQuantizationClipScaling(16, m.QueryQuantization.clip,
			out.scores[i][0].scaling)
		scoresm := softmaxquantization.GetQuantizedMatrix(len(out.scores[i]), 1, out.scores[i])
		//to int 8
		i8Scores := softmaxquantization.RequantizeMatrixInt8(scoresm)
		out.context[i] = MulInt8(i8Values, i8Scores)
	}
	return out
}

func (m *Model) GetQKV(input QuantizedInt8Matrix) IntQKV {
	return IntQKV{
		Queries: m.Query.Forward(input),
		Keys:    m.Key.Forward(input),
		Values:  m.Value.Forward(input),
	}
}
