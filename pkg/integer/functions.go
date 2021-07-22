// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package integer

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"math"
)

type Quantization struct {
	B       int     // quantization bit precision. e.g. 32
	Clip    float32 // clipping parameter used to control the outliers
	scaling float32 // scaling factor
}

type QuantizedInt8 struct {
	value   int8    // quantized value
	scaling float32 // scaling factor. x (float) = q * scaling
}

type QuantizedInt struct {
	Value   int32   // quantized Value
	Scaling float32 // Scaling factor. x (float) = q * Scaling
}

type QuantizedInt8Matrix struct {
	Matrix  [][]int8 // quantized Matrix
	Scaling float32  // scaling factor. x (float) = q * scaling
}

type QuantizedIntMatrix struct {
	Matrix  [][]int32 // quantized Matrix
	Scaling float32   // Scaling factor. x (float) = q * Scaling
}

type ExpParameters struct {
	a          float32
	b          float32
	c          float32
	ln2        float32
	cnst       int32
	qln        int32
	qb         int32
	qc         int32
	scalingOut float32
}

func NewExpParameters(q Quantization) ExpParameters {
	a := float32(0.35815147)
	b := float32(2.70732486)
	c := float32(1.0)
	ln2 := float32(-0.6931)
	cnst := int32(30)
	qln := int32(math.Floor(float64(ln2 / q.scaling)))
	qb := int32(math.Floor(float64(b / q.scaling)))
	qc := int32(math.Floor(float64(c / (a * q.scaling * q.scaling))))
	return ExpParameters{
		a:          a,
		b:          b,
		c:          c,
		ln2:        ln2,
		cnst:       cnst,
		qln:        qln,
		qb:         qb,
		qc:         qc,
		scalingOut: a * q.scaling * q.scaling,
	}
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
	if x > q.Clip {
		x = q.Clip
	}
	if x < -q.Clip {
		x = -q.Clip
	}

	return QuantizedInt{int32(math.Round(float64(x / q.scaling))), q.scaling}
}

func (q *Quantization) QuantizeInt8(x float32) QuantizedInt8 {
	if q.B != 8 {
		panic("Quantize int8: invalid b")
	}
	if x > q.Clip {
		x = q.Clip
	}
	if x < -q.Clip {
		x = -q.Clip
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

func (q *Quantization) integerPoly2(p ExpParameters, input int32) QuantizedInt {
	//qb := int32(math.Floor(float64(b / q.scaling)))
	//qc := int32(math.Floor(float64(c / (a * q.scaling * q.scaling))))
	//scalingOut := p.a * q.scaling * q.scaling
	qOut := ((input + p.qb) * (input)) + p.qc
	return QuantizedInt{qOut, p.scalingOut}
}

func (q *Quantization) integerErf(input int32) QuantizedInt {
	a := float32(-0.28888)
	b := float32(-1.769)
	c := float32(1.0)
	var qsgn = int32(1)
	qtmp := Quantization{q.B, math.MaxFloat32, q.scaling}
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
	qOut := qsgn * qL.Value
	scalingOut := qL.Scaling
	return QuantizedInt{qOut, scalingOut}
}

func (q *Quantization) IntegerGelu(input int32) QuantizedInt {
	qtmp := Quantization{q.B, math.MaxFloat32, q.scaling / 1.4142135624}
	qErf := qtmp.integerErf(input)
	qOne := int32(math.Floor(float64(1.0 / qErf.Scaling)))
	qOut := input * (qErf.Value + qOne)
	scalingOut := q.scaling * qErf.Scaling / 2

	return QuantizedInt{qOut, scalingOut}
}

func (q *Quantization) IntegerExp(input int32, p ExpParameters) QuantizedInt {
	//a := float32(0.35815147)
	//b := float32(2.70732486)
	//c := float32(1.0)
	//ln2 := float32(-0.6931)
	//cnst := int32(30)
	//qln := int32(math.Floor(float64(ln2 / q.scaling)))
	qint := input
	if input < (p.cnst * p.qln) {
		qint = p.cnst * p.qln
	}
	qp := qint / p.qln
	r := qint - p.qln*qp
	qtmp := Quantization{q.B, math.MaxFloat32, q.scaling}
	expInt := qtmp.integerPoly2(p, r)
	t := expInt.Value >> qp
	return QuantizedInt{t, expInt.Scaling}
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

func (q *Quantization) IntSoftmax(input []int32, p ExpParameters) []QuantizedInt {
	max := max(input)
	sum := int32(0)
	exp := make([]QuantizedInt, 0)
	for i := 0; i < len(input); i++ {
		exp = append(exp, q.IntegerExp(input[i]-max, p))
		sum += exp[i].Value
	}
	factor := exp[0].Scaling
	for i := 0; i < len(input); i++ {
		div := (float32(exp[i].Value) / float32(sum)) / factor
		exp[i].Value = int32(math.Floor(float64(div)))
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
	avg = avg / int32(len(input))
	stdDev := int32(0)
	for i := 0; i < len(input); i++ {
		stdDev += (input[i] - avg) * (input[i] - avg)
	}
	stdDev = stdDev / int32(len(input))
	stdDev = IntSqrt(stdDev)
	if stdDev == 0 {
		stdDev = 1
	}
	for i := 0; i < len(input); i++ {
		normalizedLayer = append(normalizedLayer, QuantizedInt{
			Value:   int32(math.Round(float64(input[i]-avg) / (float64(stdDev) * float64(q.scaling)))),
			Scaling: q.scaling,
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
			m[i][j] = data[k].Value
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
			m[i][j] = q.Quantize(data[k]).Value
			k++
		}
	}
	return QuantizedIntMatrix{m, q.scaling}
}

func (q *Quantization) DequantizeMatrix(input QuantizedIntMatrix) [][]float32 {
	qOut := NewQuantizationClipScaling(q.B, q.Clip, input.Scaling)
	m := make([][]float32, len(input.Matrix))
	for i := 0; i < len(input.Matrix); i++ {
		m[i] = make([]float32, len(input.Matrix[0]))
	}
	for i := 0; i < len(input.Matrix); i++ {
		for j := 0; j < len(input.Matrix[0]); j++ {
			m[i][j] = qOut.Dequantize(input.Matrix[i][j])
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
	m := intZeroMatrix(len(a.Matrix[0]), len(a.Matrix))
	for i := 0; i < len(a.Matrix); i++ {
		for j := 0; j < len(a.Matrix[0]); j++ {
			m[j][i] += a.Matrix[i][j]

		}
	}
	return QuantizedIntMatrix{m, a.Scaling}
}

func Mul(a, b QuantizedIntMatrix) QuantizedIntMatrix {
	if len(a.Matrix[0]) != len(b.Matrix) {
		panic("Mul: matrices with not compatible size")
	}
	m := intZeroMatrix(len(a.Matrix), len(b.Matrix[0]))
	for i := 0; i < len(a.Matrix); i++ {
		for j := 0; j < len(b.Matrix[0]); j++ {
			for k := 0; k < len(b.Matrix); k++ {
				m[i][j] += a.Matrix[i][k] * b.Matrix[k][j]
			}
		}
	}
	return QuantizedIntMatrix{m, a.Scaling * b.Scaling}
}

func Prod(a, b QuantizedIntMatrix) QuantizedIntMatrix {
	if len(a.Matrix[0]) != len(b.Matrix[0]) && (len(a.Matrix) != len(b.Matrix)) {
		panic("Prod: matrices with not compatible size")
	}
	m := intZeroMatrix(len(a.Matrix), len(b.Matrix[0]))
	for i := 0; i < len(a.Matrix); i++ {
		for j := 0; j < len(a.Matrix[0]); j++ {
			m[i][j] = a.Matrix[i][j] * b.Matrix[i][j]
		}
	}
	return QuantizedIntMatrix{m, a.Scaling * b.Scaling}
}

func ProdScalar(a QuantizedIntMatrix, scalar QuantizedInt) QuantizedIntMatrix {
	m := intZeroMatrix(len(a.Matrix), len(a.Matrix[0]))
	for i := 0; i < len(a.Matrix); i++ {
		for j := 0; j < len(a.Matrix[0]); j++ {
			m[i][j] = a.Matrix[i][j] * scalar.Value
		}
	}
	return QuantizedIntMatrix{m, a.Scaling * scalar.Scaling}
}

func Add(a, b QuantizedIntMatrix) QuantizedIntMatrix {
	if len(a.Matrix[0]) != len(b.Matrix[0]) && (len(a.Matrix) != len(b.Matrix)) {
		panic("Add: matrices with not compatible size")
	}
	if a.Scaling != b.Scaling {
		panic("Add: warning, different scaling factor")
	}
	m := intZeroMatrix(len(a.Matrix), len(a.Matrix[0]))
	for i := 0; i < len(a.Matrix); i++ {
		for j := 0; j < len(a.Matrix[0]); j++ {
			m[i][j] = a.Matrix[i][j] + b.Matrix[i][j]
		}
	}
	return QuantizedIntMatrix{m, a.Scaling}
}

func Sub(a, b QuantizedIntMatrix) QuantizedIntMatrix {
	if len(a.Matrix[0]) != len(b.Matrix[0]) && (len(a.Matrix) != len(b.Matrix)) {
		panic("Add: matrices with not compatible size")
	}
	if a.Scaling != b.Scaling {
		panic("Add: warning, different scaling factor")
	}
	m := intZeroMatrix(len(a.Matrix), len(a.Matrix[0]))
	for i := 0; i < len(a.Matrix); i++ {
		for j := 0; j < len(a.Matrix[0]); j++ {
			m[i][j] = a.Matrix[i][j] - b.Matrix[i][j]
		}
	}
	return QuantizedIntMatrix{m, a.Scaling}
}

func SubScalar(a QuantizedIntMatrix, scalar QuantizedInt) QuantizedIntMatrix {
	m := intZeroMatrix(len(a.Matrix), len(a.Matrix[0]))
	for i := 0; i < len(a.Matrix); i++ {
		for j := 0; j < len(a.Matrix[0]); j++ {
			m[i][j] = a.Matrix[i][j] - scalar.Value
		}
	}
	return QuantizedIntMatrix{m, a.Scaling}
}

func ReduceMean(a QuantizedIntMatrix) QuantizedInt {
	sum := int32(0)
	l := int32(0)
	for i := 0; i < len(a.Matrix); i++ {
		for j := 0; j < len(a.Matrix[0]); j++ {
			sum += a.Matrix[i][j]
			l++
		}
	}
	sum = sum / l
	return QuantizedInt{sum, a.Scaling}
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
	qOut := NewQuantizationClipScaling(q.B, q.Clip, input.Scaling)
	m := make([][]float32, len(input.Matrix))
	for i := 0; i < len(input.Matrix); i++ {
		m[i] = make([]float32, len(input.Matrix[0]))
	}
	for i := 0; i < len(input.Matrix); i++ {
		for j := 0; j < len(input.Matrix[0]); j++ {
			m[i][j] = qOut.DequantizeInt8(input.Matrix[i][j])
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
	m := int8ZeroMatrix(len(a.Matrix[0]), len(a.Matrix))
	for i := 0; i < len(a.Matrix); i++ {
		for j := 0; j < len(a.Matrix[0]); j++ {
			m[j][i] += a.Matrix[i][j]

		}
	}
	return QuantizedInt8Matrix{m, a.Scaling}
}

func MulInt8(a, b QuantizedInt8Matrix) QuantizedIntMatrix {
	if len(a.Matrix[0]) != len(b.Matrix) {
		panic("MulInt8: matrices with not compatible size")
	}
	m := intZeroMatrix(len(a.Matrix), len(b.Matrix[0]))
	for i := 0; i < len(a.Matrix); i++ {
		for j := 0; j < len(b.Matrix[0]); j++ {
			for k := 0; k < len(b.Matrix); k++ {
				m[i][j] += int32(a.Matrix[i][k]) * int32(b.Matrix[k][j])
			}
		}
	}
	return QuantizedIntMatrix{m, a.Scaling * b.Scaling}
}

func ProdInt8(a, b QuantizedInt8Matrix) QuantizedIntMatrix {
	if len(a.Matrix[0]) != len(b.Matrix[0]) && (len(a.Matrix) != len(b.Matrix)) {
		panic("ProdInt8: matrices with not compatible size")
	}
	m := intZeroMatrix(len(a.Matrix), len(b.Matrix[0]))
	for i := 0; i < len(a.Matrix); i++ {
		for j := 0; j < len(a.Matrix[0]); j++ {
			m[i][j] = int32(a.Matrix[i][j]) * int32(b.Matrix[i][j])
		}
	}
	return QuantizedIntMatrix{m, a.Scaling * b.Scaling}
}

func ProdScalarInt8(a QuantizedInt8Matrix, scalar QuantizedInt8) QuantizedIntMatrix {
	m := intZeroMatrix(len(a.Matrix), len(a.Matrix[0]))
	for i := 0; i < len(a.Matrix); i++ {
		for j := 0; j < len(a.Matrix[0]); j++ {
			m[i][j] = int32(a.Matrix[i][j]) * int32(scalar.value)
		}
	}
	return QuantizedIntMatrix{m, a.Scaling * scalar.scaling}
}

func AddInt8(a, b QuantizedInt8Matrix) QuantizedIntMatrix {
	if len(a.Matrix[0]) != len(b.Matrix[0]) && (len(a.Matrix) != len(b.Matrix)) {
		panic("Add: matrices with not compatible size")
	}
	if a.Scaling != b.Scaling {
		panic("Add: warning, different scaling factor")
	}
	m := intZeroMatrix(len(a.Matrix), len(a.Matrix[0]))
	for i := 0; i < len(a.Matrix); i++ {
		for j := 0; j < len(a.Matrix[0]); j++ {
			m[i][j] = int32(a.Matrix[i][j]) + int32(b.Matrix[i][j])
		}
	}
	return QuantizedIntMatrix{m, a.Scaling}
}

func SubInt8(a, b QuantizedInt8Matrix) QuantizedIntMatrix {
	if len(a.Matrix[0]) != len(b.Matrix[0]) && (len(a.Matrix) != len(b.Matrix)) {
		panic("Add: matrices with not compatible size")
	}
	if a.Scaling != b.Scaling {
		panic("Add: warning, different scaling factor")
	}
	m := intZeroMatrix(len(a.Matrix), len(a.Matrix[0]))
	for i := 0; i < len(a.Matrix); i++ {
		for j := 0; j < len(a.Matrix[0]); j++ {
			m[i][j] = int32(a.Matrix[i][j]) - int32(b.Matrix[i][j])
		}
	}
	return QuantizedIntMatrix{m, a.Scaling}
}

func SubScalarInt8(a QuantizedInt8Matrix, scalar QuantizedInt) QuantizedIntMatrix {
	m := intZeroMatrix(len(a.Matrix), len(a.Matrix[0]))
	for i := 0; i < len(a.Matrix); i++ {
		for j := 0; j < len(a.Matrix[0]); j++ {
			m[i][j] = int32(a.Matrix[i][j]) - scalar.Value
		}
	}
	return QuantizedIntMatrix{m, a.Scaling}
}

func (q *Quantization) RequantizeMatrixInt8(input QuantizedIntMatrix) QuantizedInt8Matrix {
	qOut := NewQuantization(8, q.Clip)
	m := make([][]int8, len(input.Matrix))
	for i := 0; i < len(input.Matrix); i++ {
		m[i] = make([]int8, len(input.Matrix[0]))
	}
	for i := 0; i < len(input.Matrix); i++ {
		for j := 0; j < len(input.Matrix[0]); j++ {
			m[i][j] = q.RequantizeInt8(input.Matrix[i][j], &qOut).value
		}
	}
	return QuantizedInt8Matrix{m, qOut.scaling}
}

func (q *Quantization) RequantizeMatrix(input QuantizedIntMatrix, b int) QuantizedIntMatrix {
	qOut := NewQuantization(b, q.Clip)
	m := make([][]int32, len(input.Matrix))
	for i := 0; i < len(input.Matrix); i++ {
		m[i] = make([]int32, len(input.Matrix[0]))
	}
	for i := 0; i < len(input.Matrix); i++ {
		for j := 0; j < len(input.Matrix[0]); j++ {
			dq := q.Dequantize(input.Matrix[i][j])
			m[i][j] = qOut.Quantize(dq).Value
		}
	}
	return QuantizedIntMatrix{m, qOut.scaling}
}

// Stack row vectors only
func (q *Quantization) Stack(input ...QuantizedIntMatrix) QuantizedIntMatrix {
	m := make([][]int32, len(input))
	vlen := len(input[0].Matrix[0])
	for i := 0; i < len(input); i++ {
		m[i] = make([]int32, vlen)
	}
	for i := 0; i < len(input); i++ {
		for j := 0; j < vlen; j++ {
			m[i][j] = input[i].Matrix[0][j]
		}
	}
	return QuantizedIntMatrix{m, q.scaling}
}

func (q *Quantization) StackInt8(input ...QuantizedInt8Matrix) QuantizedInt8Matrix {
	m := make([][]int8, len(input))
	vlen := len(input[0].Matrix[0])
	for i := 0; i < len(input); i++ {
		m[i] = make([]int8, vlen)
	}
	for i := 0; i < len(input); i++ {
		for j := 0; j < vlen; j++ {
			m[i][j] = input[i].Matrix[0][j]
		}
	}
	return QuantizedInt8Matrix{m, q.scaling}
}

// Concat row vectors only
func (q *Quantization) ConcatRow(input ...QuantizedIntMatrix) QuantizedIntMatrix {
	m := make([][]int32, 1)
	vlen := 0
	for i := 0; i < len(input); i++ {
		vlen += len(input[0].Matrix[0])
	}
	m[0] = make([]int32, vlen)
	k := 0
	for i := 0; i < len(input); i++ {
		for j := 0; j < len(input[i].Matrix[0]); j++ {
			m[0][k] = input[i].Matrix[0][j]
			k++
		}
	}
	return QuantizedIntMatrix{m, q.scaling}
}

func (q *Quantization) ConcatRowInt8(input ...QuantizedInt8Matrix) QuantizedInt8Matrix {
	m := make([][]int8, 1)
	vlen := 0
	for i := 0; i < len(input); i++ {
		vlen += len(input[0].Matrix[0])
	}
	m[0] = make([]int8, vlen)
	k := 0
	for i := 0; i < len(input); i++ {
		for j := 0; j < len(input[i].Matrix[0]); j++ {
			m[0][k] = input[i].Matrix[0][j]
			k++
		}
	}
	return QuantizedInt8Matrix{m, q.scaling}
}

// Concat column vectors only
func (q *Quantization) ConcatCol(input ...QuantizedIntMatrix) QuantizedIntMatrix {
	length := len(input) * len(input[0].Matrix)
	m := make([][]int32, length)
	for i := 0; i < length; i++ {
		m[i] = make([]int32, 1)
	}
	k := 0
	for i := 0; i < len(input); i++ {
		for j := 0; j < len(input[i].Matrix); j++ {
			m[k][0] = input[i].Matrix[j][0]
			k++
		}
	}
	return QuantizedIntMatrix{m, q.scaling}
}

func (q *Quantization) ConcatColInt8(input ...QuantizedInt8Matrix) QuantizedInt8Matrix {
	length := len(input) * len(input[0].Matrix)
	m := make([][]int8, length)
	for i := 0; i < length; i++ {
		m[i] = make([]int8, 1)
	}
	k := 0
	for i := 0; i < len(input); i++ {
		for j := 0; j < len(input[i].Matrix); j++ {
			m[k][0] = input[i].Matrix[j][0]
			k++
		}
	}
	return QuantizedInt8Matrix{m, q.scaling}
}
