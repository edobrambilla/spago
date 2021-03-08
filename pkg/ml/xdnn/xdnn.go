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
type XDnnModel struct {
	// Mode returns whether the model is being used for training or inference.
	Mode ProcessingMode `json:"processing_mode"`
	// Number of prototypes
	Classes []*XDnnClass `json:"classes"`
	// Labels description
	ID2Label []string `json:"id2label"`
}

// Define dataset Classes, contains features that describe prototypes and other values
type XDnnClass struct {
	//Samples           int
	PrototypesSupport []int // Number of members associated to prototypes in this class
	PrototypesVectors []*mat32.Dense
	Radius            []float32 // Degree of similarity between two vectors
	// Number of prototypes
	Prototypes int `json:"prototypes"`
	// Global mean per class
	Mean *mat32.Dense `json:"mean"`
	// Global sum squared norm per class
	SumSquaredNorm float32 `json:"norm"`
}

func NewDefaultxDNN(nClasses int, labels []string) *XDnnModel {
	if nClasses < 2 {
		panic("At least 2 classes required")
	}
	c := make([]*XDnnClass, nClasses)
	return &XDnnModel{
		Mode:     Training,
		Classes:  c,
		ID2Label: labels,
	}
}

func NewxDNNClass(vector *mat32.Dense) *XDnnClass {
	prototypesSupport := make([]int, 0)
	prototypesSupport = append(prototypesSupport, 1)
	prototypesVectors := make([]*mat32.Dense, 0)
	prototypesVectors = append(prototypesVectors, vector)
	radiusValues := make([]float32, 0)
	radiusValues = append(radiusValues, 1.30057568)
	return &XDnnClass{
		//PrototypesID:      prototypesID,
		PrototypesVectors: prototypesVectors,
		Radius:            radiusValues,
		Prototypes:        1,
		Mean:              vector,
		SumSquaredNorm:    SquaredNorm(vector),
	}
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

func Variance(vector *mat32.Dense) float32 {
	sum := float32(0.0)
	sumsqr := float32(0.0)
	for _, v := range vector.Data() {
		sum += v
		sumsqr += v * v
	}
	sumsqr *= 1.0 / float32(len(vector.Data()))
	sum *= 1.0 / float32(len(vector.Data()))
	return sumsqr - (sum * sum)
}

func (x XDnnModel) DensityIncremental(vector *mat32.Dense, index int, class int) float32 {
	var incrementalMean *mat32.Dense
	var dividedVector *mat32.Dense
	var incrementalEuclideanNorm float32
	findex := float32(index)
	f := findex / (findex + 1.0)
	r := 1.0 / (findex + 1.0)
	incrementalMean = x.Classes[class].Mean.ProdScalar(mat32.NewScalar(f).Scalar()).(*mat32.Dense)
	dividedVector = vector.ProdScalar(mat32.NewScalar(r).Scalar()).(*mat32.Dense)
	incrementalMean = incrementalMean.Add(dividedVector).(*mat32.Dense)
	incrementalEuclideanNorm = (f * x.Classes[class].SumSquaredNorm) + (r * SquaredNorm(vector))
	diffSquaredNorm := SquaredNorm(vector.Sub(incrementalMean).(*mat32.Dense))
	incrMeanSquaredNorm := SquaredNorm(incrementalMean)
	x.Classes[class].Mean = incrementalMean
	x.Classes[class].SumSquaredNorm = incrementalEuclideanNorm
	return 1.0 / (1.0 + (diffSquaredNorm + incrementalEuclideanNorm - incrMeanSquaredNorm))
}

func (x XDnnModel) Density(vector *mat32.Dense, class int) float32 {
	mean := x.Classes[class].Mean
	euclideanNorm := x.Classes[class].SumSquaredNorm
	diffSquaredNorm := SquaredNorm(vector.Sub(mean).(*mat32.Dense))
	incrMeanSquaredNorm := SquaredNorm(mean)
	return 1.0 / (1.0 + (diffSquaredNorm + euclideanNorm - incrMeanSquaredNorm))
}

func SquaredNorm(vector *mat32.Dense) float32 {
	sum := float32(0.0)
	for i := 0; i < vector.Size(); i++ {
		sum = sum + (vector.Data()[i] * vector.Data()[i])
	}
	return sum
}

func Norm(vector *mat32.Dense) float32 {
	sum := float32(0.0)
	for i := 0; i < vector.Size(); i++ {
		sum = sum + (vector.Data()[i] * vector.Data()[i])
	}
	return mat32.Sqrt(sum)
}

func Similarity(vectorA *mat32.Dense, vectorB *mat32.Dense) float32 {
	s := Variance(vectorB)
	return 1.0 / (1.0 + (SquaredNorm(vectorA.Sub(vectorB).(*mat32.Dense)) / s))
}

type maxminPair struct {
	max float32
	min float32
}

func (x XDnnModel) getMaxMinPrototype(class int) maxminPair {
	r := maxminPair{
		max: float32(-math.MaxFloat32),
		min: float32(math.MaxFloat32),
	}
	for _, p := range x.Classes[class].PrototypesVectors {
		density := x.Density(p, class)
		if density > r.max {
			r.max = density
		}
		if density < r.min {
			r.min = density
		}
	}
	return r
}

func (x XDnnModel) GetNearestPrototype(vector *mat32.Dense, class int) int {
	argmin := 0
	minNorm := float32(math.MaxFloat32)
	for j, p := range x.Classes[class].PrototypesVectors {
		norm := Norm(p.Sub(vector).(*mat32.Dense))
		if norm < minNorm {
			argmin = j
		}
	}
	return argmin
}

func (x XDnnModel) CheckExample(vector *mat32.Dense, index int, class int) {
	sampleDensity := x.DensityIncremental(vector, index, class)
	prototypesDensity := x.getMaxMinPrototype(class)
	nearestPrototypeIndex := x.GetNearestPrototype(vector, class)
	if (sampleDensity >= prototypesDensity.max) || (sampleDensity <= prototypesDensity.min) {
		x.AddDataCloud(vector, class)
	} else {
		x.UpdateDatacloud(vector, nearestPrototypeIndex, class)
	}
}

func (x XDnnModel) AddDataCloud(vector *mat32.Dense, class int) {
	x.Classes[class].PrototypesVectors = append(x.Classes[class].PrototypesVectors, vector)
	x.Classes[class].PrototypesSupport = append(x.Classes[class].PrototypesSupport, 1)
	x.Classes[class].Radius = append(x.Classes[class].Radius, 1.30057568)
}

func (x XDnnModel) UpdateDatacloud(vector *mat32.Dense, prototypeIndex int, class int) {
	prototypeNorm := SquaredNorm(x.Classes[class].PrototypesVectors[prototypeIndex])
	curSupport := x.Classes[class].PrototypesSupport[prototypeIndex]
	f := float32(curSupport / (curSupport + 1.0))
	upPrototype := x.Classes[class].PrototypesVectors[prototypeIndex].ProdScalar(f)
	upExample := vector.ProdScalar(f)
	squaredRadius := x.Classes[class].Radius[prototypeIndex] * x.Classes[class].Radius[prototypeIndex]
	upRadius := mat32.Sqrt((squaredRadius + 1.0 - prototypeNorm) / 2.0)
	x.Classes[class].PrototypesSupport[prototypeIndex] += 1
	x.Classes[class].PrototypesVectors[prototypeIndex] = upPrototype.Add(upExample).(*mat32.Dense)
	x.Classes[class].Radius[prototypeIndex] = upRadius
}
