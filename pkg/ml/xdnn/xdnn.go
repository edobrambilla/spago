// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xdnn

import "github.com/nlpodyssey/spago/pkg/mat32"

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
