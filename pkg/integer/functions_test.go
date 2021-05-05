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
	assert.Equal(t, a, 9)
	x := q.Dequantize(a)
	assert.InDelta(t, x, 3.5678, 1.0e-1)
}

func TestQuantiztion_IntegerGelu(t *testing.T) {
	q := NewQuantization(7, 50)
	a := q.Quantize(0.55)
	assert.Equal(t, a, 1)
	gelu := q.IntegerGelu(a)
	assert.Equal(t, gelu.q, -54)
	assert.Equal(t, gelu.scaling, float32(-0.0044071344))
	b := q.Quantize(-1)
	gelu = q.IntegerGelu(b)
	assert.Equal(t, gelu.q, 48)
	assert.Equal(t, gelu.scaling, float32(-0.0044071344))
}

func TestQuantiztion_IntegerExp(t *testing.T) {
	q := NewQuantization(12, 50)
	a := q.Quantize(-0.55 - 1.2)
	exp := q.IntegerExp(a)
	assert.InDelta(t, float32(exp.q)*exp.scaling, float32(0.17566888), 1.0e-6)
	q = NewQuantization(12, 50)
	b := q.Quantize(1.2 - 1.2)
	exp = q.IntegerExp(b)
	assert.InDelta(t, float32(exp.q)*exp.scaling, float32(0.9999778), 1.0e-6)
	q = NewQuantization(12, 50)
	c := q.Quantize(-500 - 1.2)
	exp = q.IntegerExp(c)
	assert.InDelta(t, float32(exp.q)*exp.scaling, float32(9.313019e-10), 1.0e-6)
}

func TestQuantiztion_IntegerSoftmax(t *testing.T) {
	q := NewQuantization(12, 50)
	v := []int{-45, 98, -491}
	softmax := q.IntSoftmax(v)
	s := float32(softmax[0].q) * softmax[0].scaling
	assert.InDelta(t, s, float32(0.1492918), 1.0e-6)
	s = float32(softmax[1].q) * softmax[0].scaling
	assert.InDelta(t, s, float32(0.84999186), 1.0e-6)
	s = float32(softmax[2].q) * softmax[0].scaling
	assert.InDelta(t, s, float32(0.00058734), 1.0e-6)
}

func TestQuantiztion_IntegerSquareRoot(t *testing.T) {
	sqrt := IntSqrt(40)
	assert.Equal(t, sqrt, 6)
	sqrt = IntSqrt(0)
	assert.Equal(t, sqrt, 0)
	sqrt = IntSqrt(9)
	assert.Equal(t, sqrt, 3)
	sqrt = IntSqrt(100)
	assert.Equal(t, sqrt, 10)
	sqrt = IntSqrt(90)
	assert.Equal(t, sqrt, 9)
}

func TestQuantization_IntegerLayerNorm(t *testing.T) {
	q := NewQuantization(12, 50)
	v := []int{-45, 98, -491}
	norm := q.IntNormalization(v)
	s := float32(norm[0].q) * norm[0].scaling
	assert.InDelta(t, s, float32(0.40293040), 1.0e-6)
	s = float32(norm[1].q) * norm[0].scaling
	assert.InDelta(t, s, float32(0.97680097), 1.0e-6)
	s = float32(norm[2].q) * norm[0].scaling
	assert.InDelta(t, s, float32(-1.3797313), 1.0e-6)
}

func TestQuantization_LinearMul(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int{2, -2, 4, 3, 4, -3}
	v2 := []int{2, 4, 8, 7, -1, 0}
	a := q.GetQuantizedIntMatrix(2, 3, v1)
	b := q.GetQuantizedIntMatrix(3, 2, v2)
	c := Mul(a, b)
	assert.Equal(t, c.matrix[0], []int{-16, -6})
	assert.Equal(t, c.matrix[1], []int{41, 40})
	assert.InDelta(t, c.scaling, float32(0.00014908), 1.0e-6)
}

func TestQuantization_LinearAdd(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int{2, -2, 4, 3, 4, -3}
	v2 := []int{2, 4, 8, 7, -1, 0}
	a := q.GetQuantizedIntMatrix(2, 3, v1)
	b := q.GetQuantizedIntMatrix(2, 3, v2)
	c := Add(a, b)
	assert.Equal(t, c.matrix[0], []int{4, 2, 12})
	assert.Equal(t, c.matrix[1], []int{10, 3, -3})
	assert.InDelta(t, c.scaling, float32(0.012210012), 1.0e-6)
}

func TestQuantization_LinearProd(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int{2, -2, 4, 3, 4, -3}
	v2 := []int{2, 4, 8, 7, -1, 0}
	a := q.GetQuantizedIntMatrix(2, 3, v1)
	b := q.GetQuantizedIntMatrix(2, 3, v2)
	c := Prod(a, b)
	assert.Equal(t, c.matrix[0], []int{4, -8, 32})
	assert.Equal(t, c.matrix[1], []int{21, -4, 0})
	assert.InDelta(t, c.scaling, float32(0.00014908), 1.0e-6)
}

func TestQuantization_LinearProdScalar(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int{2, -2, 4, 3, 4, -3}
	s := QuantizedInt{
		q:       2,
		scaling: q.scaling,
	}
	a := q.GetQuantizedIntMatrix(2, 3, v1)
	c := ProdScalar(a, s)
	assert.Equal(t, c.matrix[0], []int{4, -4, 8})
	assert.Equal(t, c.matrix[1], []int{6, 8, -6})
	assert.InDelta(t, c.scaling, float32(0.00014908), 1.0e-6)
}

func TestQuantization_LinearDequantize(t *testing.T) {
	q := NewQuantization(12, 50)
	v1 := []int{2, -2, 4, 3, 4, -3}
	a := q.GetQuantizedIntMatrix(2, 3, v1)
	f := q.DequantizeMatrix(a)
	assert.InDeltaSlice(t, f[0], []float32{0.024420024, -0.024420024, 0.048840048}, 1.0e-6)
	assert.InDeltaSlice(t, f[1], []float32{0.036630036, 0.048840048, -0.036630036}, 1.0e-6)
}
