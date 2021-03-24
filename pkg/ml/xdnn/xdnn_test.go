// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xdnn

import (
	"github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_Average(t *testing.T) {
	a := make([]mat32.Dense, 3)
	a[0] = *mat32.NewVecDense([]float32{-2.2, 3.4, 3.2, 2.9, -0.6, 4.3})
	a[1] = *mat32.NewVecDense([]float32{-0.3, 1.5, 0.9, -2.5, 0.4, 9.3})
	a[2] = *mat32.NewVecDense([]float32{0.0, 2.4, -0.3, -0.2, -0.1, 1.5})

	m := Average(a)

	assertSliceEqualApprox(t, []float32{
		-0.8333333, 2.4333333, 1.2666666, 0.0666666, -0.1, 5.0333333,
	}, m.Data())
}

func Test_StdDev(t *testing.T) {
	a := make([]mat32.Dense, 3)
	a[0] = *mat32.NewVecDense([]float32{-2.2, 3.4, 3.2, 2.9, -0.6, 4.3})
	a[1] = *mat32.NewVecDense([]float32{-0.3, 1.5, 0.9, -2.5, 0.4, 9.3})
	a[2] = *mat32.NewVecDense([]float32{0.0, 2.4, -0.3, -0.2, -0.1, 1.5})

	s := StdDev(a)

	assertSliceEqualApprox(t, []float32{
		0.9741092, 0.776029, 1.45220, 2.212590, 0.40824829, 3.226280,
	}, s.Data())
}

func Test_Min(t *testing.T) {
	a := make([]mat32.Dense, 3)
	a[0] = *mat32.NewVecDense([]float32{-2.2, 3.4, 3.2, 2.9, -0.6, 4.3})
	a[1] = *mat32.NewVecDense([]float32{-0.3, 1.5, 0.9, -2.5, 0.4, 9.3})
	a[2] = *mat32.NewVecDense([]float32{0.0, 2.4, -0.3, -0.2, -0.1, 1.5})

	s := Min(a)

	assertSliceEqualApprox(t, []float32{
		-2.2, 1.5, -0.3, -2.5, -0.6, 1.5,
	}, s.Data())
}

func Test_Max(t *testing.T) {
	a := make([]mat32.Dense, 3)
	a[0] = *mat32.NewVecDense([]float32{-2.2, 3.4, 3.2, 2.9, -0.6, 4.3})
	a[1] = *mat32.NewVecDense([]float32{-0.3, 1.5, 0.9, -2.5, 0.4, 9.3})
	a[2] = *mat32.NewVecDense([]float32{0.0, 2.4, -0.3, -0.2, -0.1, 1.5})

	s := Max(a)

	assertSliceEqualApprox(t, []float32{
		0.0, 3.4, 3.2, 2.9, 0.4, 9.3,
	}, s.Data())
}

func Test_Standardize(t *testing.T) {
	a := make([]mat32.Dense, 3)
	a[0] = *mat32.NewVecDense([]float32{-2.2, 3.4, 3.2, 2.9, -0.6, 4.3})
	a[1] = *mat32.NewVecDense([]float32{-0.3, 1.5, 0.9, -2.5, 0.4, 9.3})
	a[2] = *mat32.NewVecDense([]float32{0.0, 2.4, -0.3, -0.2, -0.1, 1.5})

	s := Standardize(a)

	assertSliceEqualApprox(t, []float32{
		-1.40299111, 1.245656, 1.331312, 1.280550409, -1.2247448, -0.22729989,
	}, s[0].Data())
	assertSliceEqualApprox(t, []float32{
		0.547508728, -1.202702, -0.252490, -1.16002801, 1.22474487, 1.3224721,
	}, s[1].Data())
	assertSliceEqualApprox(t, []float32{
		0.8554823885, -0.042953, -1.0788218, -0.120522, 0.0, -1.0951722,
	}, s[2].Data())
}

func Test_Normalize(t *testing.T) {
	a := make([]mat32.Dense, 3)
	a[0] = *mat32.NewVecDense([]float32{-2.2, 3.4, 3.2, 2.9, -0.6, 4.3})
	a[1] = *mat32.NewVecDense([]float32{-0.3, 1.5, 0.9, -2.5, 0.4, 9.3})
	a[2] = *mat32.NewVecDense([]float32{0.0, 2.4, -0.3, -0.2, -0.1, 1.5})

	s := Normalize(a)

	assertSliceEqualApprox(t, []float32{
		0.0, 1.0, 1.0, 1.0, 0.0, 0.358974,
	}, s[0].Data())
	assertSliceEqualApprox(t, []float32{
		0.8636363, 0.0, 0.34285714, 0.0, 1.0, 1.0,
	}, s[1].Data())
	assertSliceEqualApprox(t, []float32{
		1.0, 0.4736842, 0.0, 0.4259259, 0.5, 0.0,
	}, s[2].Data())
}

func Test_SquaredNorm(t *testing.T) {
	a := mat32.NewVecDense([]float32{-2.2, 3.4, 3.2, 2.9, -0.6, 4.3})
	s := SquaredNorm(a)
	assertEqualApprox(t, 53.9, s)
}

func Test_DensityIncremental(t *testing.T) {
	a := make([]mat32.Dense, 3)
	a[0] = *mat32.NewVecDense([]float32{-2.2, 3.4, 3.2, 2.9, -0.6, 4.3})
	a[1] = *mat32.NewVecDense([]float32{-0.3, 1.5, 0.9, -2.5, 0.4, 9.3})
	model := simpleXDNN()
	model.Classes[0] = NewxDNNClass(a[0])
	s := model.DensityIncremental(a[0], 0, 0)
	assertEqualApprox(t, 1.0, s)
	s = model.DensityIncremental(a[1], 1, 0)
	assertEqualApprox(t, 0.028706760, s)
}

func Test_Density(t *testing.T) {
	a := make([]mat32.Dense, 3)
	a[0] = *mat32.NewVecDense([]float32{-2.2, 3.4, 3.2, 2.9, -0.6, 4.3})
	a[1] = *mat32.NewVecDense([]float32{-0.3, 1.5, 0.9, -2.5, 0.4, 9.3})
	a[2] = *mat32.NewVecDense([]float32{0.0, 2.4, -0.3, -0.2, -0.1, 1.5})
	model := simpleXDNN()
	model.Classes[0] = NewxDNNClass(a[0])
	s := model.DensityIncremental(a[0], 0, 0)
	s = model.DensityIncremental(a[1], 1, 0)
	d := model.Density(a[2], 0)
	assertEqualApprox(t, 0.028706760, s)
	assertEqualApprox(t, 0.01877758, d)
}

func Test_CheckExample(t *testing.T) {
	a := make([]mat32.Dense, 5)
	a[0] = *mat32.NewVecDense([]float32{-2.2, 3.4, 3.2, 2.9, -0.6, 4.3})
	a[1] = *mat32.NewVecDense([]float32{-0.3, 1.5, 0.9, -2.5, 0.4, 9.3})
	a[2] = *mat32.NewVecDense([]float32{0.0, 2.4, -0.3, -0.2, -0.1, 1.5})
	a[3] = *mat32.NewVecDense([]float32{0.3, 2.6, -1.0, 2.8, 3.5, -1.4})
	a[4] = *mat32.NewVecDense([]float32{-0.6, -0.2, 0.5, 0.8, -3.0, -1.0})
	model := simpleXDNN()
	model.CheckExample(a[0], 0, 0, false)
	assert.Equal(t, model.Classes[0].Prototypes, 1)
	assert.Equal(t, model.Classes[0].PrototypesSupport[0], 1)
	assert.Equal(t, model.Classes[0].PrototypesVectors[0], *mat32.NewVecDense([]float32{-2.2, 3.4, 3.2, 2.9, -0.6, 4.3}))
	assert.Equal(t, model.Classes[0].Mean, *mat32.NewVecDense([]float32{-2.2, 3.4, 3.2, 2.9, -0.6, 4.3}))
	model.CheckExample(a[1], 1, 0, false)
	assert.Equal(t, model.Classes[0].Prototypes, 2)
	assert.Equal(t, model.Classes[0].PrototypesSupport[0], 1)
	assert.Equal(t, model.Classes[0].PrototypesSupport[1], 1)
	assert.Equal(t, model.Classes[0].PrototypesVectors[0], *mat32.NewVecDense([]float32{-2.2, 3.4, 3.2, 2.9, -0.6, 4.3}))
	assert.Equal(t, model.Classes[0].PrototypesVectors[1], *mat32.NewVecDense([]float32{-0.3, 1.5, 0.9, -2.5, 0.4, 9.3}))
	model.CheckExample(a[2], 2, 0, false)
	assert.Equal(t, model.Classes[0].Prototypes, 2)
	assert.Equal(t, model.Classes[0].PrototypesSupport[0], 2)
	assert.Equal(t, model.Classes[0].PrototypesSupport[1], 1)
	assertSliceEqualApprox(t, model.Classes[0].PrototypesVectors[0].Data(), []float32{-1.1, 2.9, 1.45, 1.35, -0.35, 2.9})
	assertSliceEqualApprox(t, model.Classes[0].PrototypesVectors[1].Data(), []float32{-0.3, 1.5, 0.9, -2.5, 0.4, 9.3})
	model.CheckExample(a[3], 3, 0, false)
	assert.Equal(t, model.Classes[0].Prototypes, 2)
	assert.Equal(t, model.Classes[0].PrototypesSupport[0], 3)
	assert.Equal(t, model.Classes[0].PrototypesSupport[1], 1)
	assertSliceEqualApprox(t, model.Classes[0].PrototypesVectors[0].Data(), []float32{
		-0.6333326001, 2.7999980658, 0.633332367, 1.8333324324, 0.9333335655, 1.4666647338,
	})
	assertSliceEqualApprox(t, model.Classes[0].PrototypesVectors[1].Data(), []float32{-0.3, 1.5, 0.9, -2.5, 0.4, 9.3})
	model.CheckExample(a[4], 4, 0, false)
	assert.Equal(t, model.Classes[0].Prototypes, 2)
	assert.Equal(t, model.Classes[0].PrototypesSupport[0], 4)
	assert.Equal(t, model.Classes[0].PrototypesSupport[1], 1)
	assertSliceEqualApprox(t, model.Classes[0].PrototypesVectors[0].Data(), []float32{
		-0.625, 2.05, 0.6, 1.575, -0.049999, 0.85,
	})
	assertSliceEqualApprox(t, model.Classes[0].PrototypesVectors[1].Data(), []float32{-0.3, 1.5, 0.9, -2.5, 0.4, 9.3})
}

func assertEqualApprox(t *testing.T, expected, actual float32) {
	t.Helper()
	assert.InDelta(t, expected, actual, 1.0e-04)
}

func assertSliceEqualApprox(t *testing.T, expected, actual []float32) {
	t.Helper()
	assert.InDeltaSlice(t, expected, actual, 1.0e-04)
}

func simpleXDNN() *XDnnModel {
	labels := []string{"001", "002", "003"}
	return NewDefaultxDNN(3, labels)
}
