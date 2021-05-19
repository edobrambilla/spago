// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package integer

type IntQKV struct {
	Queries QuantizedIntMatrix
	Keys    QuantizedIntMatrix
	Values  QuantizedIntMatrix
}
