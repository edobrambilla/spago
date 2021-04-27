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
