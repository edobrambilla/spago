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

type QuantizedInt struct {
	q       int
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

func (q *Quantization) integerPoly(a, b, c float32, input int) QuantizedInt {
	qb := int(math.Floor(float64(b / q.scaling)))
	qc := int(math.Floor(float64(c / (a * q.scaling * q.scaling))))
	scalingOut := a * q.scaling * q.scaling
	qOut := ((input + qb) * (input + qb)) + qc
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
