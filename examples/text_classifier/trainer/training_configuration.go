// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trainer

type TrainingConfig struct {
	Seed             uint64
	BatchSize        int
	Epochs           int
	GradientClipping float64
	TrainCorpusPath  string
	EvalCorpusPath   string
	ModelPath        string
	IncludeTitle     bool
	IncludeBody      bool
	LabelsMap        map[string]int
}
