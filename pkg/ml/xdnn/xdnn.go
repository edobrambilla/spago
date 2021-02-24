// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xdnn

import (
	"github.com/nlpodyssey/spago/pkg/mat32"
	"math"
)

const (
	// Training is to be used during the training phase of a model.
	Training ProcessingMode = iota
	// Inference
	Inference
)

// ProcessingMode regulates the different usage of some operations
// depending on whether you're doing training or inference.
type ProcessingMode int

// xDnn Model
type xDnnModel struct {
	// Mode returns whether the model is being used for training or inference.
	Mode ProcessingMode `json:"processing_mode"`
	// Number of prototypes
	Prototypes int `json:"prototypes"`
	// Global mean
	Mean mat32.Dense `json:"mean"`
	// Labels description
	ID2Label map[int]string `json:"id2label"`
}

// Define dataset Classes, contains features that describe prototypes and other values
type xDnnClass struct {
	Samples           []mat32.Dense
	PrototypesID      int // todo names?
	PrototypesVectors []mat32.Dense
	Support           int     // Number of members associated to this class
	Radius            float32 // Degree of similarity between two vectors
}

func Standardize(vectors []*mat32.Dense) []*mat32.Dense {
	ret := make([]*mat32.Dense, len(vectors))
	average := Average(vectors)
	stdev := StdDev(vectors)
	for i, v := range vectors {
		sub := v.Sub(average).(*mat32.Dense)
		div := sub.Div(stdev).(*mat32.Dense)
		ret[i] = div
	}
	return ret
}

func Normalize(vectors []*mat32.Dense) []*mat32.Dense {
	ret := make([]*mat32.Dense, len(vectors))
	standardizedVectors := Standardize(vectors)
	min := Min(standardizedVectors)
	max := Max(standardizedVectors)
	diff := max.Sub(min)
	for i, v := range standardizedVectors {
		sub := v.Sub(min).(*mat32.Dense)
		div := sub.Div(diff).(*mat32.Dense)
		ret[i] = div
	}
	return ret
}

func Min(vectors []*mat32.Dense) *mat32.Dense {
	minVector := mat32.NewInitDense(vectors[0].Rows(), vectors[0].Columns(), math.MaxFloat32)
	for _, v := range vectors {
		for j := 0; j < minVector.Size(); j++ {
			if minVector.Data()[j] > v.Data()[j] {
				minVector.Data()[j] = v.Data()[j]
			}
		}
	}
	return minVector
}

func Max(vectors []*mat32.Dense) *mat32.Dense {
	maxvector := mat32.NewInitDense(vectors[0].Rows(), vectors[0].Columns(), -math.MaxFloat32)
	for _, v := range vectors {
		for j := 0; j < maxvector.Size(); j++ {
			if maxvector.Data()[j] < v.Data()[j] {
				maxvector.Data()[j] = v.Data()[j]
			}
		}
	}
	return maxvector
}

func Average(vectors []*mat32.Dense) *mat32.Dense {
	sum := mat32.NewEmptyVecDense(vectors[0].Size())
	for _, v := range vectors {
		for j := 0; j < sum.Size(); j++ {
			sum.Data()[j] += v.Data()[j]
		}
	}
	return sum.ProdScalar(mat32.NewScalar(1.0 / float32(len(vectors))).Scalar()).(*mat32.Dense)
}

func StdDev(vectors []*mat32.Dense) *mat32.Dense {
	sum := mat32.NewEmptyVecDense(vectors[0].Size())
	sumsqr := mat32.NewEmptyVecDense(vectors[0].Size())
	for _, v := range vectors {
		for j := 0; j < sum.Size(); j++ {
			sum.Data()[j] += v.Data()[j]
			sumsqr.Data()[j] += v.Data()[j] * v.Data()[j]
		}
	}
	sumsqr.ProdScalarInPlace(mat32.NewScalar(1.0 / float32(len(vectors))).Scalar())
	sum.ProdScalarInPlace(mat32.NewScalar(1.0 / float32(len(vectors))).Scalar())
	sum = sum.Prod(sum).(*mat32.Dense)
	diff := sumsqr.Sub(sum)
	//diff = diff.ProdScalar(mat32.NewScalar(1.0 / float32(len(vectors))).Scalar())
	sqrt := diff.Sqrt()
	return sqrt.(*mat32.Dense)
}
