// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hashing

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"gonum.org/v1/gonum/floats"
	"testing"
)

func TestModel_Forward(t *testing.T) {

	h2 := New(5, 5, 2)

	// == Forward

	x2 := mat.NewVecDense([]float64{-0.8, -0.9, -0.9, 1.0, 0.5})
	y2 := h2.GetHash(x2)

	if !floats.EqualApprox(y2.Data(), []float64{1.0, 0.0, 1.0, 0.0, 0.0}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}

	h3 := New(5, 5, 3)

	// == Forward

	x3 := mat.NewVecDense([]float64{-0.8, -0.9, -0.9, 1.0, 0.5})
	y3 := h3.GetHash(x3)

	if !floats.EqualApprox(y3.Data(), []float64{0.0, -1.0, 1.0, 0.0, 0.0}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}
}
