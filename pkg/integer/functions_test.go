// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package integer

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestQuantization_Quantize(t *testing.T) {
	q := NewQuantization(7, 50)
	a := q.Quantize(3.5678)
	assert.Equal(t, a.Value, int32(9))
	x := q.Dequantize(a.Value)
	assert.InDelta(t, x, 3.5678, 1.0e-1)
}

func TestQuantiztion_IntegerGelu(t *testing.T) {
	q := NewQuantization(7, 50)
	a := q.Quantize(0.55)
	assert.Equal(t, a.Value, int32(1))
	gelu := q.IntegerGelu(a.Value)
	assert.Equal(t, gelu.Value, int32(-54))
	assert.Equal(t, gelu.Scaling, float32(-0.0044071344))
	b := q.Quantize(-1)
	gelu = q.IntegerGelu(b.Value)
	assert.Equal(t, gelu.Value, int32(48))
	assert.Equal(t, gelu.Scaling, float32(-0.0044071344))
}

func TestQuantiztion_IntegerExp(t *testing.T) {
	q := NewQuantization(12, 50)
	a := q.Quantize(-0.55 - 1.2)
	exp := q.IntegerExp(a.Value)
	assert.InDelta(t, float32(exp.Value)*exp.Scaling, float32(0.17566888), 1.0e-6)
	q = NewQuantization(12, 50)
	b := q.Quantize(1.2 - 1.2)
	exp = q.IntegerExp(b.Value)
	assert.InDelta(t, float32(exp.Value)*exp.Scaling, float32(0.9999778), 1.0e-6)
	q = NewQuantization(12, 50)
	c := q.Quantize(-500 - 1.2)
	exp = q.IntegerExp(c.Value)
	assert.InDelta(t, float32(exp.Value)*exp.Scaling, float32(9.313019e-10), 1.0e-6)
}

func TestQuantiztion_IntegerSoftmax(t *testing.T) {
	q := NewQuantization(12, 50)
	v := []int32{-45, 98, -491}
	softmax := q.IntSoftmax(v)
	s := float32(softmax[0].Value) * softmax[0].Scaling
	assert.InDelta(t, s, float32(0.1492918), 1.0e-6)
	s = float32(softmax[1].Value) * softmax[0].Scaling
	assert.InDelta(t, s, float32(0.84999186), 1.0e-6)
	s = float32(softmax[2].Value) * softmax[0].Scaling
	assert.InDelta(t, s, float32(0.00058734), 1.0e-6)
}

func TestQuantiztion_IntegerSquareRoot(t *testing.T) {
	sqrt := IntSqrt(40)
	assert.Equal(t, sqrt, int32(6))
	sqrt = IntSqrt(0)
	assert.Equal(t, sqrt, int32(0))
	sqrt = IntSqrt(9)
	assert.Equal(t, sqrt, int32(3))
	sqrt = IntSqrt(100)
	assert.Equal(t, sqrt, int32(10))
	sqrt = IntSqrt(90)
	assert.Equal(t, sqrt, int32(9))
}

func TestQuantization_IntegerLayerNorm(t *testing.T) {
	q := NewQuantization(12, 50)
	v := []int32{-45, 98, -491}
	norm := q.IntNormalization(v)
	s := float32(norm[0].Value) * norm[0].Scaling
	assert.InDelta(t, s, float32(0.40293040), 1.0e-6)
	s = float32(norm[1].Value) * norm[0].Scaling
	assert.InDelta(t, s, float32(0.97680097), 1.0e-6)
	s = float32(norm[2].Value) * norm[0].Scaling
	assert.InDelta(t, s, float32(-1.3797313), 1.0e-6)
}

func TestQuantization_Transpose(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int32{2, -2, 4, 3, 4, -3}
	a := q.GetQuantizedMatrixFromInt(2, 3, v1)
	c := Transpose(a)
	assert.Equal(t, c.Matrix[0], []int32{2, 3})
	assert.Equal(t, c.Matrix[1], []int32{-2, 4})
	assert.Equal(t, c.Matrix[2], []int32{4, -3})
	assert.InDelta(t, c.Scaling, float32(0.012210012), 1.0e-6)
}

func TestQuantization_LinearMul(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int32{2, -2, 4, 3, 4, -3}
	v2 := []int32{2, 4, 8, 7, -1, 0}
	a := q.GetQuantizedMatrixFromInt(2, 3, v1)
	b := q.GetQuantizedMatrixFromInt(3, 2, v2)
	c := Mul(a, b)
	assert.Equal(t, c.Matrix[0], []int32{-16, -6})
	assert.Equal(t, c.Matrix[1], []int32{41, 40})
	assert.InDelta(t, c.Scaling, float32(0.00014908), 1.0e-6)
}

func TestQuantization_LinearAdd(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int32{2, -2, 4, 3, 4, -3}
	v2 := []int32{2, 4, 8, 7, -1, 0}
	a := q.GetQuantizedMatrixFromInt(2, 3, v1)
	b := q.GetQuantizedMatrixFromInt(2, 3, v2)
	c := Add(a, b)
	assert.Equal(t, c.Matrix[0], []int32{4, 2, 12})
	assert.Equal(t, c.Matrix[1], []int32{10, 3, -3})
	assert.InDelta(t, c.Scaling, float32(0.012210012), 1.0e-6)
}

func TestQuantization_LinearSub(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int32{2, -2, 4, 3, 4, -3}
	v2 := []int32{2, 4, 8, 7, -1, 0}
	a := q.GetQuantizedMatrixFromInt(2, 3, v1)
	b := q.GetQuantizedMatrixFromInt(2, 3, v2)
	c := Sub(a, b)
	assert.Equal(t, c.Matrix[0], []int32{0, -6, -4})
	assert.Equal(t, c.Matrix[1], []int32{-4, 5, -3})
	assert.InDelta(t, c.Scaling, float32(0.012210012), 1.0e-6)
}

func TestQuantization_LinearProd(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int32{2, -2, 4, 3, 4, -3}
	v2 := []int32{2, 4, 8, 7, -1, 0}
	a := q.GetQuantizedMatrixFromInt(2, 3, v1)
	b := q.GetQuantizedMatrixFromInt(2, 3, v2)
	c := Prod(a, b)
	assert.Equal(t, c.Matrix[0], []int32{4, -8, 32})
	assert.Equal(t, c.Matrix[1], []int32{21, -4, 0})
	assert.InDelta(t, c.Scaling, float32(0.00014908), 1.0e-6)
}

func TestQuantization_LinearProdScalar(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int32{2, -2, 4, 3, 4, -3}
	s := QuantizedInt{
		Value:   2,
		Scaling: q.scaling,
	}
	a := q.GetQuantizedMatrixFromInt(2, 3, v1)
	c := ProdScalar(a, s)
	assert.Equal(t, c.Matrix[0], []int32{4, -4, 8})
	assert.Equal(t, c.Matrix[1], []int32{6, 8, -6})
	assert.InDelta(t, c.Scaling, float32(0.00014908), 1.0e-6)
}

func TestQuantization_LinearSubScalarInt(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int32{2, -2, 4, 3, 4, -3}
	s := QuantizedInt{
		Value:   2,
		Scaling: q.scaling,
	}
	a := q.GetQuantizedMatrixFromInt(2, 3, v1)
	c := SubScalar(a, s)
	assert.Equal(t, c.Matrix[0], []int32{0, -4, 2})
	assert.Equal(t, c.Matrix[1], []int32{1, 2, -5})
	assert.InDelta(t, c.Scaling, float32(0.012210012), 1.0e-6)
}

func TestQuantization_Reducemean(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int32{2, 2, 4, 3, 4, -2}
	a := q.GetQuantizedMatrixFromInt(2, 3, v1)
	c := ReduceMean(a)
	assert.Equal(t, c.Value, int32(2))
	assert.InDelta(t, c.Scaling, float32(0.012210012), 1.0e-6)
}

func TestQuantization_LinearDequantize(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int32{2, -2, 4, 3, 4, -3}
	a := q.GetQuantizedMatrixFromInt(2, 3, v1)
	f := q.DequantizeMatrix(a)
	assert.InDeltaSlice(t, f[0], []float32{0.024420024, -0.024420024, 0.048840048}, 1.0e-6)
	assert.InDeltaSlice(t, f[1], []float32{0.036630036, 0.048840048, -0.036630036}, 1.0e-6)
}

//int8

func TestQuantization_LinearMulInt8(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int8{2, -2, 4, 3, 4, -3}
	v2 := []int8{2, 4, 8, 7, -1, 0}
	a := q.GetQuantizedMatrixFromInt8(2, 3, v1)
	b := q.GetQuantizedMatrixFromInt8(3, 2, v2)
	c := MulInt8(a, b)
	assert.Equal(t, c.Matrix[0], []int32{-16, -6})
	assert.Equal(t, c.Matrix[1], []int32{41, 40})
	assert.InDelta(t, c.Scaling, float32(0.00014908), 1.0e-6)
}

func TestQuantization_LinearAddInt8(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int8{2, -2, 4, 3, 4, -3}
	v2 := []int8{2, 4, 8, 7, -1, 0}
	a := q.GetQuantizedMatrixFromInt8(2, 3, v1)
	b := q.GetQuantizedMatrixFromInt8(2, 3, v2)
	c := AddInt8(a, b)
	assert.Equal(t, c.Matrix[0], []int32{4, 2, 12})
	assert.Equal(t, c.Matrix[1], []int32{10, 3, -3})
	assert.InDelta(t, c.Scaling, float32(0.012210012), 1.0e-6)
}

func TestQuantization_LinearSubInt8(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int8{2, -2, 4, 3, 4, -3}
	v2 := []int8{2, 4, 8, 7, -1, 0}
	a := q.GetQuantizedMatrixFromInt8(2, 3, v1)
	b := q.GetQuantizedMatrixFromInt8(2, 3, v2)
	c := SubInt8(a, b)
	assert.Equal(t, c.Matrix[0], []int32{0, -6, -4})
	assert.Equal(t, c.Matrix[1], []int32{-4, 5, -3})
	assert.InDelta(t, c.Scaling, float32(0.012210012), 1.0e-6)
}

func TestQuantization_LinearProdInt8(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int8{2, -2, 4, 3, 4, -3}
	v2 := []int8{2, 4, 8, 7, -1, 0}
	a := q.GetQuantizedMatrixFromInt8(2, 3, v1)
	b := q.GetQuantizedMatrixFromInt8(2, 3, v2)
	c := ProdInt8(a, b)
	assert.Equal(t, c.Matrix[0], []int32{4, -8, 32})
	assert.Equal(t, c.Matrix[1], []int32{21, -4, 0})
	assert.InDelta(t, c.Scaling, float32(0.00014908), 1.0e-6)
}

func TestQuantization_LinearProdScalarInt8(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int8{2, -2, 4, 3, 4, -3}
	s := QuantizedInt8{
		value:   2,
		scaling: q.scaling,
	}
	a := q.GetQuantizedMatrixFromInt8(2, 3, v1)
	c := ProdScalarInt8(a, s)
	assert.Equal(t, c.Matrix[0], []int32{4, -4, 8})
	assert.Equal(t, c.Matrix[1], []int32{6, 8, -6})
	assert.InDelta(t, c.Scaling, float32(0.00014908), 1.0e-6)
}

func TestQuantization_LinearSubScalarInt8(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int8{2, -2, 4, 3, 4, -3}
	s := QuantizedInt{
		Value:   2,
		Scaling: q.scaling,
	}
	a := q.GetQuantizedMatrixFromInt8(2, 3, v1)
	c := SubScalarInt8(a, s)
	assert.Equal(t, c.Matrix[0], []int32{0, -4, 2})
	assert.Equal(t, c.Matrix[1], []int32{1, 2, -5})
	assert.InDelta(t, c.Scaling, float32(0.012210012), 1.0e-6)
}

func TestQuantization_Requantize(t *testing.T) {
	q := NewQuantization(32, 50)
	q8 := NewQuantization(8, 50)
	a := q.Quantize(0.1)
	qi := a.Value
	assert.Equal(t, qi, int32(8589935))
	x := q.RequantizeInt8(qi, &q8)
	assert.Equal(t, x.value, int8(1))
	assert.InDelta(t, x.scaling, 0.19607843, 1.0e-6)
}

func TestQuantization_Stack(t *testing.T) {
	q := NewQuantization(8, 50)
	v1 := []int32{2, -2, 4, 3, 4, -3}
	v2 := []int32{2, -2, 4, 3, 9, -3}
	v3 := []int32{2, -2, 7, 3, 0, -3}
	a1 := q.GetQuantizedMatrixFromInt(1, 6, v1)
	a2 := q.GetQuantizedMatrixFromInt(1, 6, v2)
	a3 := q.GetQuantizedMatrixFromInt(1, 6, v3)
	c := q.Stack(a1, a2, a3)
	assert.Equal(t, c.Matrix[2][4], int32(0))
}

func TestQuantization_StackInt8(t *testing.T) {
	q := NewQuantization(8, 50)
	v1 := []int8{2, -2, 4, 3, 4, -3}
	v2 := []int8{2, -2, 4, 3, 9, -3}
	v3 := []int8{2, -2, 7, 3, 0, -3}
	a1 := q.GetQuantizedMatrixFromInt8(1, 6, v1)
	a2 := q.GetQuantizedMatrixFromInt8(1, 6, v2)
	a3 := q.GetQuantizedMatrixFromInt8(1, 6, v3)
	c := q.StackInt8(a1, a2, a3)
	assert.Equal(t, c.Matrix[2][2], int8(7))
}

func TestQuantization_Concat(t *testing.T) {
	q := NewQuantization(8, 50)
	v1 := []int32{2, -2, 4, 3, 4, -3}
	v2 := []int32{2, -2, 4, 3, 9, -3}
	v3 := []int32{2, -2, 7, 3, 0, -3}
	a1 := q.GetQuantizedMatrixFromInt(1, 6, v1)
	a2 := q.GetQuantizedMatrixFromInt(1, 6, v2)
	a3 := q.GetQuantizedMatrixFromInt(1, 6, v3)
	c := q.ConcatRow(a1, a2, a3)
	assert.Equal(t, c.Matrix[0][16], int32(0))
}

func TestQuantization_ConcatInt8(t *testing.T) {
	q := NewQuantization(8, 50)
	v1 := []int8{2, -2, 4, 3, 4, -3}
	v2 := []int8{2, -2, 4, 3, 9, -3}
	v3 := []int8{2, -2, 7, 3, 0, -3}
	a1 := q.GetQuantizedMatrixFromInt8(1, 6, v1)
	a2 := q.GetQuantizedMatrixFromInt8(1, 6, v2)
	a3 := q.GetQuantizedMatrixFromInt8(1, 6, v3)
	c := q.ConcatRowInt8(a1, a2, a3)
	assert.Equal(t, c.Matrix[0][14], int8(7))
}

func TestQuantization_ConcatCol(t *testing.T) {
	q := NewQuantization(8, 50)
	v1 := []int32{2, -2, 4, 3, 4, -3}
	v2 := []int32{2, -2, 4, 3, 9, -3}
	v3 := []int32{2, -2, 7, 3, 0, -3}
	a1 := q.GetQuantizedMatrixFromInt(6, 1, v1)
	a2 := q.GetQuantizedMatrixFromInt(6, 1, v2)
	a3 := q.GetQuantizedMatrixFromInt(6, 1, v3)
	c := q.ConcatCol(a1, a2, a3)
	assert.Equal(t, c.Matrix[16][0], int32(0))
}

func TestQuantization_ConcatColInt8(t *testing.T) {
	q := NewQuantization(8, 50)
	v1 := []int8{2, -2, 4, 3, 4, -3}
	v2 := []int8{2, -2, 4, 3, 9, -3}
	v3 := []int8{2, -2, 7, 3, 0, -3}
	a1 := q.GetQuantizedMatrixFromInt8(6, 1, v1)
	a2 := q.GetQuantizedMatrixFromInt8(6, 1, v2)
	a3 := q.GetQuantizedMatrixFromInt8(6, 1, v3)
	c := q.ConcatColInt8(a1, a2, a3)
	assert.Equal(t, c.Matrix[14][0], int8(7))
}
