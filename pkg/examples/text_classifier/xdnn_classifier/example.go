// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trainer

import (
	"encoding/json"
	"fmt"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/basetokenizer"
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

func GetTokenizedExample(e Example, includeTitle, includeBody bool) []string {
	out := []string{}
	if includeTitle {
		tokenized := Tokenize(e.Title)
		if len(tokenized) > 0 {
			out = append(out, tokenized...)
		}
	}
	if includeBody {
		tokenized := Tokenize(e.Body)
		if len(tokenized) > 0 {
			out = append(out, tokenized...)
		}
	}
	return out
}

func PadTokens(tokens []string, n int) []string {
	length := len(tokens)

	for i := 0; i < n-length; i++ {
		tokens = append(tokens, "<EOS>")
	}
	return tokens
}

func Tokenize(text string) []string {
	tokenizer := basetokenizer.New()
	return tokenizers.GetStrings(tokenizer.Tokenize(text))
}
