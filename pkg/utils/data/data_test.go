// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package data

import (
	"testing"
)

func Equal(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

func TestGenerateNGrams(t *testing.T) {

	n := GenerateNGrams(3, 5)

	if !Equal(n[0], []int{0, 1, 2}) {
		t.Error("Ngrams error")
	}
	if !Equal(n[1], []int{1, 2, 3}) {
		t.Error("Ngrams error")
	}
	if !Equal(n[2], []int{2, 3, 4}) {
		t.Error("Ngrams error")
	}

	n1 := GenerateNGrams(4, 4)

	if !Equal(n1[0], []int{0, 1, 2, 3}) {
		t.Error("Ngrams error")
	}

	n2 := GenerateNGrams(5, 4)

	if !(n2 == nil) {
		t.Error("Ngrams error")
	}

	n3 := GenerateNGrams(1, 3)

	if !Equal(n3[0], []int{0}) {
		t.Error("Ngrams error")
	}
	if !Equal(n3[1], []int{1}) {
		t.Error("Ngrams error")
	}
	if !Equal(n3[2], []int{2}) {
		t.Error("Ngrams error")
	}
}
