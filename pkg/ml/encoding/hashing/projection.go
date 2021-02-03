// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hashing

import (
	"github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"math"
)

type Config struct {
	// The encoder output size n
	OutputSize int
	// Narity 2: Output encoder is Rn ∈ {0, 1}.
	// Narity 3: Output encoder is Rn ∈ {-1, 0, 1}.
	Narity int
}

type Data struct {
	// Random vectors, normal distributed
	W []mat32.Matrix
	Config
}

// Hashing scheme like Charikar, Similarity Estimation Techniques from Rounding Algorithms
func New(inputSize int, hashSize int, narity int) *Data {
	if narity > 3 || narity < 2 {
		panic("projection: narity can be 2 or 3.")
	}
	w := make([]mat32.Matrix, hashSize)
	rndGen := rand.NewLockedRand(33)
	for i := 0; i < hashSize; i++ {
		w[i] = mat32.NewEmptyVecDense(inputSize)
		initializers.Normal(w[i], 0, 1, rndGen)
	}
	return &Data{
		W:      w,
		Config: Config{hashSize, narity},
	}
}

// Simple scheme, todo can be improved
func quantization(n float64, narity int, min float64, max float64) float64 {
	if narity > 3 || narity < 2 {
		panic("projection: narity can be 2 or 3.")
	}

	if narity == 2 {
		if n > 0 {
			return 1.0
		} else {
			return 0.0
		}
	}
	delta := max - (math.Abs(max-min) / float64(narity))
	if narity == 3 {
		if n > delta {
			return 1.0
		} else if math.Abs(n) <= delta {
			return 0.0
		} else if n < -delta {
			return -1.0
		}
	}
	return 0.0
}

func (d *Data) GetHashDot(input mat32.Matrix) *mat32.Dense {
	out := mat32.NewEmptyVecDense(d.OutputSize)
	for i := 0; i < d.OutputSize; i++ {
		p := input.DotUnitary(d.W[i])
		out.Data()[i] = float32(quantization(float64(p), d.Config.Narity, -3.0, 3.0))
	}
	return out
}

func (d *Data) GetHashProjection(input mat32.Matrix) *mat32.Dense {
	out := mat32.NewEmptyVecDense(d.OutputSize)
	k := 0
	for i := 0; i < d.OutputSize; i += 2 {
		if input.Data()[i] == 0.0 && input.Data()[i+1] == 0.0 {
			out.Data()[k] = -1.0
		} else if input.Data()[i] == 1.0 && input.Data()[i+1] == 1.0 {
			out.Data()[k] = 1.0
		} else {
			out.Data()[k] = 0.0
		}
		k++
	}
	return out
}
