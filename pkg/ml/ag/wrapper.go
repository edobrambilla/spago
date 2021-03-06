// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag/fn"
)

var (
	_ fn.Operand = &wrapper{}
	_ GradValue  = &wrapper{}
	_ Node       = &wrapper{}
)

type wrapper struct {
	GradValue
	graph    *Graph
	timeStep int
	id       int
	wrapGrad bool
}

// ID returns the ID of the node in the graph.
func (r *wrapper) ID() int {
	return r.id
}

// Graph returns the graph this node belongs to.
func (r *wrapper) Graph() *Graph {
	return r.graph
}

// Grad returns the gradients accumulated during the backward pass.
func (r *wrapper) Grad() mat.Matrix {
	if !r.wrapGrad {
		return nil
	}
	return r.GradValue.Grad()
}

// PropagateGrad propagates the gradients to the node.
func (r *wrapper) PropagateGrad(gx mat.Matrix) {
	if !r.wrapGrad {
		return
	}
	r.GradValue.PropagateGrad(gx)
}

// HasGrad returns true if there are accumulated gradients.
func (r *wrapper) HasGrad() bool {
	if !r.wrapGrad {
		return false
	}
	return r.GradValue.HasGrad()
}

// RequiresGrad returns true if the node requires gradients.
func (r *wrapper) RequiresGrad() bool {
	if !r.wrapGrad {
		return false
	}
	return r.GradValue.RequiresGrad()
}

// ZeroGrad set the gradients to zeros.
func (r *wrapper) ZeroGrad() {
	if !r.wrapGrad {
		return
	}
	r.GradValue.ZeroGrad()
}

func (r *wrapper) TimeStep() int {
	return r.timeStep
}
