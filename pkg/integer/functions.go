// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package integer

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"math"
)

type Quantization struct {
	b       int     // quantization bit precision. e.g. 32
	clip    float32 // clipping parameter used to control the outliers
	scaling float32 // scaling factor
}

type QuantizedInt struct {
	q       int     // quantized value
	scaling float32 // scaling factor. x (float) = q * scaling
}

type QuantizedIntMatrix struct {
	matrix  [][]int // quantized matrix
	scaling float32 // scaling factor. x (float) = q * scaling
}

func NewQuantization(b int, clip float32) Quantization {
	scaling := clip / (mat.Pow(2.0, float32(b)) - 1)
	return Quantization{b, clip, scaling}
}

func NewQuantizationScaling(b int, scaling float32) Quantization {
	clip := scaling * (mat.Pow(2.0, float32(b)) - 1)
	return Quantization{b, clip, scaling}
}

func NewQuantizationClipScaling(b int, clip float32, scaling float32) Quantization {
	return Quantization{b, clip, scaling}
}

func (q *Quantization) Quantize(x float32) int {
	if x > q.clip {
		x = q.clip
	}
	if x < -q.clip {
		x = -q.clip
	}
	return int(math.Round(float64(x / q.scaling)))
}

func (q *Quantization) Dequantize(x int) float32 {
	return float32(x) * q.scaling
}

func (q *Quantization) integerPoly(a, b, c float32, input int) QuantizedInt {
	qb := int(math.Floor(float64(b / q.scaling)))
	qc := int(math.Floor(float64(c / (a * q.scaling * q.scaling))))
	scalingOut := a * q.scaling * q.scaling
	qOut := ((input + qb) * (input + qb)) + qc
	return QuantizedInt{qOut, scalingOut}
}

func (q *Quantization) integerPoly2(a, b, c float32, input int) QuantizedInt {
	qb := int(math.Floor(float64(b / q.scaling)))
	qc := int(math.Floor(float64(c / (a * q.scaling * q.scaling))))
	scalingOut := a * q.scaling * q.scaling
	qOut := ((input + qb) * (input)) + qc
	return QuantizedInt{qOut, scalingOut}
}

func (q *Quantization) integerErf(input int) QuantizedInt {
	a := float32(-0.28888)
	b := float32(-1.769)
	c := float32(1.0)
	var qsgn = 1
	qtmp := Quantization{q.b, math.MaxFloat32, q.scaling}
	if input > 0 {
		if input > (int(-b / q.scaling)) {
			input = int(-b / q.scaling)
		}
	} else {
		qsgn = -1
		if -input > (int(-b / q.scaling)) {
			input = -int(-b / q.scaling)
		} else {
			input = -input
		}
	}
	qL := qtmp.integerPoly(a, b, c, input)
	qOut := qsgn * qL.q
	scalingOut := qL.scaling
	return QuantizedInt{qOut, scalingOut}
}

func (q *Quantization) IntegerGelu(input int) QuantizedInt {
	qtmp := Quantization{q.b, math.MaxFloat32, q.scaling / 1.4142135624}
	qErf := qtmp.integerErf(input)
	qOne := int(math.Floor(float64(1.0 / qErf.scaling)))
	qOut := input * (qErf.q + qOne)
	scalingOut := q.scaling * qErf.scaling / 2

	return QuantizedInt{qOut, scalingOut}
}

func (q *Quantization) IntegerExp(input int) QuantizedInt {
	a := float32(0.35815147)
	b := float32(2.70732486)
	c := float32(1.0)
	ln2 := float32(-0.6931)
	cnst := 30
	qln := int(math.Floor(float64(ln2 / q.scaling)))
	qint := input
	if input < (cnst * qln) {
		qint = cnst * qln
	}
	qp := int(math.Floor(float64(qint / qln)))
	r := qint - qln*qp
	qtmp := Quantization{q.b, math.MaxFloat32, q.scaling}
	expInt := qtmp.integerPoly2(a, b, c, r)
	t := expInt.q >> qp
	return QuantizedInt{t, expInt.scaling}
}

func max(input []int) int {
	var m int
	for i, e := range input {
		if i == 0 || e > m {
			m = e
		}
	}
	return m
}

func (q *Quantization) IntSoftmax(input []int) []QuantizedInt {
	max := max(input)
	sum := 0
	exp := make([]QuantizedInt, 0)
	for i := 0; i < len(input); i++ {
		exp = append(exp, q.IntegerExp(input[i]-max))
		sum += exp[i].q
	}
	factor := exp[0].scaling
	for i := 0; i < len(input); i++ {
		div := (float32(exp[i].q) / float32(sum)) / factor
		exp[i].q = int(math.Floor(float64(div)))
	}
	return exp
}

func bitCount(input int) int {
	if input == 0 {
		return 0
	}
	return int(math.Log2(float64(input)) + 1)
}

func IntSqrt(input int) int {
	if input == 0 {
		return 0
	}
	if input < 0 {
		panic("IntegerSqrt: input cannot be negative.")
	}
	b := float64(bitCount(input))
	x := math.Pow(2.0, math.Ceil(b/2))
	i := 0
	for {
		y := math.Floor((x + math.Floor(float64(input)/x)) / 2)
		if y >= x {
			return int(x)
		}
		x = y
		i++
	}
}

func (q *Quantization) IntNormalization(input []int) []QuantizedInt {
	normalizedLayer := make([]QuantizedInt, 0)
	avg := 0
	for i := 0; i < len(input); i++ {
		avg += input[i]
	}
	avg = int(math.Round(float64(avg) / float64(len(input))))
	stdDev := 0
	for i := 0; i < len(input); i++ {
		stdDev += (input[i] - avg) * (input[i] - avg)
	}
	stdDev = int(math.Round(float64(stdDev) / float64(len(input))))
	stdDev = IntSqrt(stdDev)
	for i := 0; i < len(input); i++ {
		normalizedLayer = append(normalizedLayer, QuantizedInt{
			q:       int(math.Round(float64(input[i]-avg) / (float64(stdDev) * float64(q.scaling)))),
			scaling: q.scaling,
		})
	}
	return normalizedLayer
}

func (q *Quantization) GetQuantizedIntMatrix(rows, cols int, data []int) QuantizedIntMatrix {
	m := make([][]int, rows)
	for i := 0; i < rows; i++ {
		m[i] = make([]int, cols)
	}
	k := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m[i][j] = data[k]
			k++
		}
	}
	return QuantizedIntMatrix{m, q.scaling}
}

func (q *Quantization) QuantizeFloatMatrix(rows, cols int, data []float32) QuantizedIntMatrix {
	m := make([][]int, rows)
	for i := 0; i < rows; i++ {
		m[i] = make([]int, cols)
	}
	k := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m[i][j] = q.Quantize(data[k])
			k++
		}
	}
	return QuantizedIntMatrix{m, q.scaling}
}

func (q *Quantization) DequantizeMatrix(input QuantizedIntMatrix) [][]float32 {
	qOut := NewQuantizationScaling(q.b, input.scaling)
	m := make([][]float32, len(input.matrix))
	for i := 0; i < len(input.matrix); i++ {
		m[i] = make([]float32, len(input.matrix[0]))
	}
	for i := 0; i < len(input.matrix); i++ {
		for j := 0; j < len(input.matrix[0]); j++ {
			m[i][j] = qOut.Dequantize(input.matrix[i][j])
		}
	}
	return m
}

func intZeroMatrix(rows, cols int) [][]int {
	m := make([][]int, rows)
	for i := 0; i < rows; i++ {
		m[i] = make([]int, cols)
	}
	return m
}

func Mul(a, b QuantizedIntMatrix) QuantizedIntMatrix {
	if len(a.matrix[0]) != len(b.matrix) {
		panic("mat32: matrices with not compatible size")
	}
	m := intZeroMatrix(len(a.matrix), len(b.matrix[0]))
	for i := 0; i < len(a.matrix); i++ {
		for j := 0; j < len(b.matrix[0]); j++ {
			for k := 0; k < len(b.matrix); k++ {
				m[i][j] += a.matrix[i][k] * b.matrix[k][j]
			}
		}
	}
	return QuantizedIntMatrix{m, a.scaling * b.scaling}
}

func Prod(a, b QuantizedIntMatrix) QuantizedIntMatrix {
	if len(a.matrix[0]) != len(b.matrix[0]) && (len(a.matrix) != len(b.matrix)) {
		panic("mat32: matrices with not compatible size")
	}
	m := intZeroMatrix(len(a.matrix), len(b.matrix[0]))
	for i := 0; i < len(a.matrix); i++ {
		for j := 0; j < len(a.matrix[0]); j++ {
			m[i][j] = a.matrix[i][j] * b.matrix[i][j]
		}
	}
	return QuantizedIntMatrix{m, a.scaling * b.scaling}
}

func ProdScalar(a QuantizedIntMatrix, scalar QuantizedInt) QuantizedIntMatrix {
	m := intZeroMatrix(len(a.matrix), len(a.matrix[0]))
	for i := 0; i < len(a.matrix); i++ {
		for j := 0; j < len(a.matrix[0]); j++ {
			m[i][j] = a.matrix[i][j] * scalar.q
		}
	}
	return QuantizedIntMatrix{m, a.scaling * scalar.scaling}
}

func Add(a, b QuantizedIntMatrix) QuantizedIntMatrix {
	if len(a.matrix[0]) != len(b.matrix[0]) && (len(a.matrix) != len(b.matrix)) {
		panic("mat32: matrices with not compatible size")
	}
	if a.scaling != b.scaling {
		panic("Add: warning, different scaling factor")
	}
	m := intZeroMatrix(len(a.matrix), len(a.matrix[0]))
	for i := 0; i < len(a.matrix); i++ {
		for j := 0; j < len(a.matrix[0]); j++ {
			m[i][j] = a.matrix[i][j] + b.matrix[i][j]
		}
	}
	return QuantizedIntMatrix{m, a.scaling}
}
