// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trainer

import (
	"bufio"
	"fmt"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"github.com/nlpodyssey/spago/pkg/nlp/classifiers/prado"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
	"io"
	"os"
)

type TrainingConfig struct {
	Seed             uint64
	BatchSize        int
	Epochs           int
	GradientClipping float64
	TrainCorpusPath  string
	ModelPath        string
}

type Trainer struct {
	TrainingConfig
	randGen       *rand.LockedRand
	optimizer     *gd.GradientDescent
	bestLoss      float64
	lastBatchLoss float64
	model         *prado.Model
	countLine     int
}

func NewTrainer(model *prado.Model, config TrainingConfig, optimizer *gd.GradientDescent) *Trainer {
	return &Trainer{
		TrainingConfig: config,
		randGen:        rand.NewLockedRand(config.Seed),
		optimizer:      optimizer,
		model:          model,
	}
}

func (t *Trainer) Train() {
	t.forEachLine(func(i int, text string) {
		//t.trainPassage(text)
		e := GetExample(text)
		println(e.Category)
		tokenized := t.tokenize(e.Title)
		println(tokenized)
		t.optimizer.IncBatch()
		t.optimizer.IncExample()
		t.optimizer.Optimize()

		t.countLine++
	})
}

func (t *Trainer) tokenize(text string) []string {
	tokenizer := wordpiecetokenizer.New(t.model.Vocabulary)
	tokenized := append(tokenizers.GetStrings(tokenizer.Tokenize(text)), wordpiecetokenizer.DefaultSequenceSeparator)
	return append([]string{wordpiecetokenizer.DefaultClassToken}, tokenized...)
}

func (t *Trainer) forEachLine(callback func(i int, line string)) (err error) {
	file, err := os.Open(t.TrainCorpusPath)
	if err != nil {
		return err
	}
	defer file.Close()

	// Start reading from the file with a reader.
	reader := bufio.NewReader(file)
	var line string
	i := 0
	for {
		line, err = reader.ReadString('\n')
		if err != nil && err != io.EOF {
			break
		}

		// Process the line here
		callback(i, line)
		if err != nil {
			break
		}
	}
	if err != io.EOF {
		fmt.Printf(" > Failed with error: %v\n", err)
		return err
	}
	return
}
