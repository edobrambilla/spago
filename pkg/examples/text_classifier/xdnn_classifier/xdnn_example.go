// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trainer

import "github.com/nlpodyssey/spago/pkg/mat32"

type xDNNExample struct {
	Category      int         `json:"category"`
	TokenizedText []string    `json:"text"`
	BiRNNVector   mat32.Dense `json:"birnnvector"`
}
