// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trainer

import (
	"encoding/json"
	"fmt"
)

type Example struct {
	Category string `json:"category"`
	Title    string `json:"title"`
	Body     string `json:"body"`
}

func GetExample(s string) Example {
	var data Example
	err := json.Unmarshal([]byte(s), &data)
	if err != nil {
		fmt.Printf("%+v\n", data)
	}
	return data
}
