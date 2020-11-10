// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hashing

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"math"
)

type Config struct {
	OutputSize int
	Narity     int
}

type Data struct {
	W []mat.Matrix
	Config
}

func New(inputSize int, hashSize int, narity int) *Data {
	if narity > 3 || narity < 2 {
		panic("projection: narity can be 2 or 3.")
	}
	w := make([]mat.Matrix, hashSize)
	rndGen := rand.NewLockedRand(33)
	for i := 0; i < hashSize; i++ {
		w[i] = mat.NewEmptyVecDense(inputSize)
		initializers.Normal(w[i], 0, 1, rndGen)
	}
	return &Data{
		W:      w,
		Config: Config{hashSize, narity},
	}
}

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
	delta := math.Abs(max-min) / float64(narity)
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

func (d *Data) GetHash(input mat.Matrix) *mat.Dense {
	out := mat.NewEmptyVecDense(d.OutputSize)
	for i := 0; i < d.OutputSize; i++ {
		p := input.DotUnitary(d.W[i])
		out.Data()[i] = quantization(p, d.Config.Narity, -1.0, 1.0)
	}
	return out
}
