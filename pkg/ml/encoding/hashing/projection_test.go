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

	h := New(5, 5, 3)

	// == Forward

	x := mat.NewVecDense([]float64{-0.8, -0.9, -0.9, 1.0, 0.5})
	y := h.GetHash(x)

	if !floats.EqualApprox(y.Data(), []float64{-0.456097, -0.855358, -0.79552, 0.844718}, 1.0e-05) {
		t.Error("The output doesn't match the expected values")
	}
}
