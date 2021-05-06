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
func NewFrom(m *selfattention.Model) *Model {
	qq := NewQuantization(12, 50)
	qk := NewQuantization(12, 50)
	qv := NewQuantization(12, 50)
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

func (m *Model) Forward(input QuantizedIntMatrix) Output {
	out := Output{
		scores:  make([][]QuantizedInt, len(input.matrix[0])),
		context: make([]QuantizedIntMatrix, len(input.matrix[0])),
	}
	qkv := m.GetQKV(input)
	tQueries := Transpose(qkv.Queries)
	scores := Mul(tQueries, qkv.Keys)
	qkquantization := NewQuantizationClipScaling(m.QueryQuantization.b, m.QueryQuantization.scaling, scores.scaling)
	scaledScores := ProdScalar(scores, qkquantization.Quantize(m.ScaleFactor))
	scoresquantization := NewQuantizationClipScaling(m.QueryQuantization.b, m.QueryQuantization.scaling, scaledScores.scaling)
	for i, attscores := range scaledScores.matrix {
		out.scores[i] = scoresquantization.IntSoftmax(attscores)
		scoresm := scoresquantization.GetQuantizedMatrix(1, len(out.scores[i]), out.scores[i])
		out.context[i] = Mul(scoresm, qkv.Values)
	}
	return out
}

func (m *Model) GetQKV(input QuantizedIntMatrix) IntQKV {
	return IntQKV{
		Queries: m.Query.Forward(input),
		Keys:    m.Key.Forward(input),
		Values:  m.Value.Forward(input),
	}
}
