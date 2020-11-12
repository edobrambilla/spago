// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package data

import (
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/utils"
)

// GenerateBatches generates a list of batches so that the classes distribution among them is approximately the same.
// The class is given by the callback for each i-th element up to size.
// The size of each batch depends on number of classes (batchFactor * nClasses).
// Each batch consists in a list of indices.
func GenerateBatches(size, batchFactor int, class func(i int) int) [][]int {
	groupsByClass := make(map[int][]int)
	for i := 0; i < size; i++ {
		c := class(i)
		groupsByClass[c] = append(groupsByClass[c], i)
	}
	nClasses := len(groupsByClass)
	batchSize := batchFactor * nClasses
	batchList := make([][]int, 0)
	for k := 0; k < size; k++ {
		if k%batchSize == 0 {
			batchList = append(batchList, []int{})
		}
	}
	distribution := make([]float64, nClasses)
	for i := 0; i < nClasses; i++ {
		distribution[i] = float64(len(groupsByClass[i])) / float64(size)
	}
	k := 0
	for k < size {
		class := rand.WeightedChoice(distribution) // this uses the global random
		if len(groupsByClass[class]) > 0 {
			var exampleIndex int
			exampleIndex, groupsByClass[class] = groupsByClass[class][0], groupsByClass[class][1:] // pop
			index := k % len(batchList)
			batchList[index] = append(batchList[index], exampleIndex)
			k++
		}
	}
	return batchList
}

// ForEachBatch divides the dataset into batches, returning the start-end of each batch with a callback.
// This function assumes that the dataset has already been shuffled.
func ForEachBatch(datasetSize, batchSize int, callback func(start, end int)) {
	for start := 0; start < datasetSize; start += batchSize {
		end := utils.MinInt(start+batchSize, datasetSize-1)
		callback(start, end)
	}
}

func GenerateNGrams(n, dataSize int) [][]int {
	if n <= 0 {
		panic("utils: invalid ngrams size")
	}
	var out [][]int
	if dataSize-n+1 > 0 {
		out = make([][]int, dataSize-n+1)
	}
	for i := 0; i < dataSize; i++ {
		if i < dataSize-n+1 {
			out[i] = make([]int, n)
			p := 0
			for k := i; k < i+n; k++ {
				out[i][p] = k
				p++
			}
		}

	}
	return out
}
