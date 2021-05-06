// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package integer

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
)

type IntQKV struct {
	Queries QuantizedIntMatrix
	Keys    QuantizedIntMatrix
	Values  QuantizedIntMatrix
}

func Stack(g *ag.Graph, q Quantization, xs []ag.Node) QuantizedIntMatrix {
	StackedNodes := g.Stack(xs...)
	StackedNodes = g.T(StackedNodes)
	x := len(xs[0].Value().Data())
	y := len(xs)
	return q.QuantizeFloatMatrix(x, y, StackedNodes.Value().Data())
}
