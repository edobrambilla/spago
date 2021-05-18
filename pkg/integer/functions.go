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

type QuantizedInt8 struct {
	value   int8    // quantized value
	scaling float32 // scaling factor. x (float) = q * scaling
}

type QuantizedInt struct {
	value   int32   // quantized value
	scaling float32 // scaling factor. x (float) = q * scaling
}

type QuantizedInt8Matrix struct {
	matrix  [][]int8 // quantized matrix
	scaling float32  // scaling factor. x (float) = q * scaling
}

type QuantizedIntMatrix struct {
	matrix  [][]int32 // quantized matrix
	scaling float32   // scaling factor. x (float) = q * scaling
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

func (q *Quantization) Quantize(x float32) QuantizedInt {
	if x > q.clip {
		x = q.clip
	}
	if x < -q.clip {
		x = -q.clip
	}

	return QuantizedInt{int32(math.Round(float64(x / q.scaling))), q.scaling}
}

func (q *Quantization) QuantizeInt8(x float32) QuantizedInt8 {
	if q.b != 8 {
		panic("Quantize int8: invalid b")
	}
	if x > q.clip {
		x = q.clip
	}
	if x < -q.clip {
		x = -q.clip
	}

	return QuantizedInt8{int8(math.Round(float64(x / q.scaling))), q.scaling}
}

func (q *Quantization) Dequantize(x int32) float32 {
	return float32(x) * q.scaling
}

func (q *Quantization) DequantizeInt8(x int8) float32 {
	return float32(x) * q.scaling
}

// Requantize quantized int to int8
func (q *Quantization) RequantizeInt8(x int32, qInt8 *Quantization) QuantizedInt8 {
	f := q.Dequantize(x)
	return qInt8.QuantizeInt8(f)
}

func (q *Quantization) integerPoly(a, b, c float32, input int32) QuantizedInt {
	qb := int32(math.Floor(float64(b / q.scaling)))
	qc := int32(math.Floor(float64(c / (a * q.scaling * q.scaling))))
	scalingOut := a * q.scaling * q.scaling
	qOut := ((input + qb) * (input + qb)) + qc
	return QuantizedInt{qOut, scalingOut}
}

func (q *Quantization) integerPoly2(a, b, c float32, input int32) QuantizedInt {
	qb := int32(math.Floor(float64(b / q.scaling)))
	qc := int32(math.Floor(float64(c / (a * q.scaling * q.scaling))))
	scalingOut := a * q.scaling * q.scaling
	qOut := ((input + qb) * (input)) + qc
	return QuantizedInt{qOut, scalingOut}
}

func (q *Quantization) integerErf(input int32) QuantizedInt {
	a := float32(-0.28888)
	b := float32(-1.769)
	c := float32(1.0)
	var qsgn = int32(1)
	qtmp := Quantization{q.b, math.MaxFloat32, q.scaling}
	if input > 0 {
		if input > (int32(-b / q.scaling)) {
			input = int32(-b / q.scaling)
		}
	} else {
		qsgn = -1
		if -input > (int32(-b / q.scaling)) {
			input = -int32(-b / q.scaling)
		} else {
			input = -input
		}
	}
	qL := qtmp.integerPoly(a, b, c, input)
	qOut := qsgn * qL.value
	scalingOut := qL.scaling
	return QuantizedInt{qOut, scalingOut}
}

func (q *Quantization) IntegerGelu(input int32) QuantizedInt {
	qtmp := Quantization{q.b, math.MaxFloat32, q.scaling / 1.4142135624}
	qErf := qtmp.integerErf(input)
	qOne := int32(math.Floor(float64(1.0 / qErf.scaling)))
	qOut := input * (qErf.value + qOne)
	scalingOut := q.scaling * qErf.scaling / 2

	return QuantizedInt{qOut, scalingOut}
}

func (q *Quantization) IntegerExp(input int32) QuantizedInt {
	a := float32(0.35815147)
	b := float32(2.70732486)
	c := float32(1.0)
	ln2 := float32(-0.6931)
	cnst := int32(30)
	qln := int32(math.Floor(float64(ln2 / q.scaling)))
	qint := input
	if input < (cnst * qln) {
		qint = cnst * qln
	}
	qp := int32(math.Floor(float64(qint / qln)))
	r := qint - qln*qp
	qtmp := Quantization{q.b, math.MaxFloat32, q.scaling}
	expInt := qtmp.integerPoly2(a, b, c, r)
	t := expInt.value >> qp
	return QuantizedInt{t, expInt.scaling}
}

func max(input []int32) int32 {
	var m int32
	for i, e := range input {
		if i == 0 || e > m {
			m = e
		}
	}
	return m
}

func (q *Quantization) IntSoftmax(input []int32) []QuantizedInt {
	max := max(input)
	sum := int32(0)
	exp := make([]QuantizedInt, 0)
	for i := 0; i < len(input); i++ {
		exp = append(exp, q.IntegerExp(input[i]-max))
		sum += exp[i].value
	}
	factor := exp[0].scaling
	for i := 0; i < len(input); i++ {
		div := (float32(exp[i].value) / float32(sum)) / factor
		exp[i].value = int32(math.Floor(float64(div)))
	}
	return exp
}

func bitCount(input int32) int32 {
	if input == 0 {
		return 0
	}
	return int32(math.Log2(float64(input)) + 1)
}

func IntSqrt(input int32) int32 {
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
			return int32(x)
		}
		x = y
		i++
	}
}

func (q *Quantization) IntNormalization(input []int32) []QuantizedInt {
	normalizedLayer := make([]QuantizedInt, 0)
	avg := int32(0)
	for i := 0; i < len(input); i++ {
		avg += input[i]
	}
	avg = int32(math.Round(float64(avg) / float64(len(input))))
	stdDev := int32(0)
	for i := 0; i < len(input); i++ {
		stdDev += (input[i] - avg) * (input[i] - avg)
	}
	stdDev = int32(math.Round(float64(stdDev) / float64(len(input))))
	stdDev = IntSqrt(stdDev)
	for i := 0; i < len(input); i++ {
		normalizedLayer = append(normalizedLayer, QuantizedInt{
			value:   int32(math.Round(float64(input[i]-avg) / (float64(stdDev) * float64(q.scaling)))),
			scaling: q.scaling,
		})
	}
	return normalizedLayer
}

func (q *Quantization) GetQuantizedMatrixFromInt(rows, cols int, data []int32) QuantizedIntMatrix {
	m := make([][]int32, rows)
	for i := 0; i < rows; i++ {
		m[i] = make([]int32, cols)
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

func (q *Quantization) GetQuantizedMatrix(rows, cols int, data []QuantizedInt) QuantizedIntMatrix {
	m := make([][]int32, rows)
	for i := 0; i < rows; i++ {
		m[i] = make([]int32, cols)
	}
	k := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m[i][j] = data[k].value
			k++
		}
	}
	return QuantizedIntMatrix{m, q.scaling}
}

func (q *Quantization) QuantizeFloatMatrix(rows, cols int, data []float32) QuantizedIntMatrix {
	m := make([][]int32, rows)
	for i := 0; i < rows; i++ {
		m[i] = make([]int32, cols)
	}
	k := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m[i][j] = q.Quantize(data[k]).value
			k++
		}
	}
	return QuantizedIntMatrix{m, q.scaling}
}

func (q *Quantization) DequantizeMatrix(input QuantizedIntMatrix) [][]float32 {
	qOut := NewQuantizationClipScaling(q.b, q.clip, input.scaling)
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

func intZeroMatrix(rows, cols int) [][]int32 {
	m := make([][]int32, rows)
	for i := 0; i < rows; i++ {
		m[i] = make([]int32, cols)
	}
	return m
}

func Transpose(a QuantizedIntMatrix) QuantizedIntMatrix {
	m := intZeroMatrix(len(a.matrix[0]), len(a.matrix))
	for i := 0; i < len(a.matrix); i++ {
		for j := 0; j < len(a.matrix[0]); j++ {
			m[j][i] += a.matrix[i][j]

		}
	}
	return QuantizedIntMatrix{m, a.scaling}
}

func Mul(a, b QuantizedIntMatrix) QuantizedIntMatrix {
	if len(a.matrix[0]) != len(b.matrix) {
		panic("Mul: matrices with not compatible size")
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
		panic("Prod: matrices with not compatible size")
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
			m[i][j] = a.matrix[i][j] * scalar.value
		}
	}
	return QuantizedIntMatrix{m, a.scaling * scalar.scaling}
}

func Add(a, b QuantizedIntMatrix) QuantizedIntMatrix {
	if len(a.matrix[0]) != len(b.matrix[0]) && (len(a.matrix) != len(b.matrix)) {
		panic("Add: matrices with not compatible size")
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

// Int8 functions

func (q *Quantization) GetQuantizedMatrixFromInt8(rows, cols int, data []int8) QuantizedInt8Matrix {
	m := make([][]int8, rows)
	for i := 0; i < rows; i++ {
		m[i] = make([]int8, cols)
	}
	k := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m[i][j] = data[k]
			k++
		}
	}
	return QuantizedInt8Matrix{m, q.scaling}
}

func (q *Quantization) GetQuantizedMatrixInt8(rows, cols int, data []QuantizedInt8) QuantizedInt8Matrix {
	m := make([][]int8, rows)
	for i := 0; i < rows; i++ {
		m[i] = make([]int8, cols)
	}
	k := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m[i][j] = data[k].value
			k++
		}
	}
	return QuantizedInt8Matrix{m, q.scaling}
}

func (q *Quantization) QuantizeFloatMatrixInt8(rows, cols int, data []float32) QuantizedInt8Matrix {
	m := make([][]int8, rows)
	for i := 0; i < rows; i++ {
		m[i] = make([]int8, cols)
	}
	k := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m[i][j] = q.QuantizeInt8(data[k]).value
			k++
		}
	}
	return QuantizedInt8Matrix{m, q.scaling}
}

func (q *Quantization) DequantizeMatrixInt8(input QuantizedInt8Matrix) [][]float32 {
	qOut := NewQuantizationClipScaling(q.b, q.clip, input.scaling)
	m := make([][]float32, len(input.matrix))
	for i := 0; i < len(input.matrix); i++ {
		m[i] = make([]float32, len(input.matrix[0]))
	}
	for i := 0; i < len(input.matrix); i++ {
		for j := 0; j < len(input.matrix[0]); j++ {
			m[i][j] = qOut.DequantizeInt8(input.matrix[i][j])
		}
	}
	return m
}

func int8ZeroMatrix(rows, cols int) [][]int8 {
	m := make([][]int8, rows)
	for i := 0; i < rows; i++ {
		m[i] = make([]int8, cols)
	}
	return m
}

func TransposeInt8(a QuantizedInt8Matrix) QuantizedInt8Matrix {
	m := int8ZeroMatrix(len(a.matrix[0]), len(a.matrix))
	for i := 0; i < len(a.matrix); i++ {
		for j := 0; j < len(a.matrix[0]); j++ {
			m[j][i] += a.matrix[i][j]

		}
	}
	return QuantizedInt8Matrix{m, a.scaling}
}

func MulInt8(a, b QuantizedInt8Matrix) QuantizedIntMatrix {
	if len(a.matrix[0]) != len(b.matrix) {
		panic("MulInt8: matrices with not compatible size")
	}
	m := intZeroMatrix(len(a.matrix), len(b.matrix[0]))
	for i := 0; i < len(a.matrix); i++ {
		for j := 0; j < len(b.matrix[0]); j++ {
			for k := 0; k < len(b.matrix); k++ {
				m[i][j] += int32(a.matrix[i][k]) * int32(b.matrix[k][j])
			}
		}
	}
	return QuantizedIntMatrix{m, a.scaling * b.scaling}
}

func ProdInt8(a, b QuantizedInt8Matrix) QuantizedIntMatrix {
	if len(a.matrix[0]) != len(b.matrix[0]) && (len(a.matrix) != len(b.matrix)) {
		panic("ProdInt8: matrices with not compatible size")
	}
	m := intZeroMatrix(len(a.matrix), len(b.matrix[0]))
	for i := 0; i < len(a.matrix); i++ {
		for j := 0; j < len(a.matrix[0]); j++ {
			m[i][j] = int32(a.matrix[i][j]) * int32(b.matrix[i][j])
		}
	}
	return QuantizedIntMatrix{m, a.scaling * b.scaling}
}

func ProdScalarInt8(a QuantizedInt8Matrix, scalar QuantizedInt8) QuantizedIntMatrix {
	m := intZeroMatrix(len(a.matrix), len(a.matrix[0]))
	for i := 0; i < len(a.matrix); i++ {
		for j := 0; j < len(a.matrix[0]); j++ {
			m[i][j] = int32(a.matrix[i][j]) * int32(scalar.value)
		}
	}
	return QuantizedIntMatrix{m, a.scaling * scalar.scaling}
}

func AddInt8(a, b QuantizedInt8Matrix) QuantizedIntMatrix {
	if len(a.matrix[0]) != len(b.matrix[0]) && (len(a.matrix) != len(b.matrix)) {
		panic("Add: matrices with not compatible size")
	}
	if a.scaling != b.scaling {
		panic("Add: warning, different scaling factor")
	}
	m := intZeroMatrix(len(a.matrix), len(a.matrix[0]))
	for i := 0; i < len(a.matrix); i++ {
		for j := 0; j < len(a.matrix[0]); j++ {
			m[i][j] = int32(a.matrix[i][j]) + int32(b.matrix[i][j])
		}
	}
	return QuantizedIntMatrix{m, a.scaling}
}

func (q *Quantization) RequantizeMatrixInt8(input QuantizedIntMatrix) QuantizedInt8Matrix {
	qOut := NewQuantization(8, q.clip)
	m := make([][]int8, len(input.matrix))
	for i := 0; i < len(input.matrix); i++ {
		m[i] = make([]int8, len(input.matrix[0]))
	}
	for i := 0; i < len(input.matrix); i++ {
		for j := 0; j < len(input.matrix[0]); j++ {
			m[i][j] = q.RequantizeInt8(input.matrix[i][j], &qOut).value
		}
	}
	return QuantizedInt8Matrix{m, qOut.scaling}
}

func (q *Quantization) RequantizeMatrix(input QuantizedIntMatrix, b int) QuantizedIntMatrix {
	qOut := NewQuantization(b, q.clip)
	m := make([][]int32, len(input.matrix))
	for i := 0; i < len(input.matrix); i++ {
		m[i] = make([]int32, len(input.matrix[0]))
	}
	for i := 0; i < len(input.matrix); i++ {
		for j := 0; j < len(input.matrix[0]); j++ {
			dq := q.Dequantize(input.matrix[i][j])
			m[i][j] = qOut.Quantize(dq).value
		}
	}
	return QuantizedIntMatrix{m, qOut.scaling}
}
