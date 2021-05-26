// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package integer

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_Layernorm(t *testing.T) {
	q := NewQuantization(16, 50)
	w := []float32{0.1, 0.3, 0.4}
	b := []float32{0.7, -0.2, -0.9}
	intModel := NewLayerNormIntModel(3, w, b)
	x1 := []float32{-0.8, -0.9, -0.9}
	x2 := []float32{0.8, -0.3, 0.5}
	q1 := q.QuantizeFloatMatrix(1, 3, x1)
	q2 := q.QuantizeFloatMatrix(1, 3, x2)
	c := intModel.Forward(q1, q2)
	assert.Equal(t, c[0].Matrix[0][0], int32(1447393))
	assert.Equal(t, c[0].Matrix[0][1], int32(-714972))
	assert.Equal(t, c[0].Matrix[0][2], int32(-2041321))
	assert.Equal(t, c[1].Matrix[0][0], int32(1375343))
	assert.Equal(t, c[1].Matrix[0][1], int32(-1046664))
	assert.Equal(t, c[1].Matrix[0][2], int32(-1299861))
}
