// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package integer

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"math"
)

type Quantization struct {
	b       int
	clip    float32
	scaling float32
}

func NewQuantization(b int, clip float32) Quantization {
	scaling := clip / (mat.Pow(2.0, float32(b)) - 1)
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
