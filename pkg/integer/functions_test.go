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
