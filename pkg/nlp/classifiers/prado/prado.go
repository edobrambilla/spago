// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package prado

import (
	"encoding/json"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
	"os"
)

const (
	DefaultConfigurationFile = "config.json"
	DefaultVocabularyFile    = "vocab.txt"
	DefaultModelFile         = "spago_model.bin"
	DefaultEmbeddingsStorage = "embeddings_storage"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Config struct {
	HiddenAct             string            `json:"hidden_act"`
	ConvAct               string            `json:"conv_act"`
	ProjectionSize        int               `json:"projection_sise"`
	ProjectionArity       int               `json:"projection_arity"`
	EncodingSize          int               `json:"encoding_size"`
	UnigramsChannels      int               `json:"unigrams_channels"`
	BigramsChannels       int               `json:"bigrams_channels"`
	TrigramsChannels      int               `json:"trigrams_channels"`
	FourgramsChannels     int               `json:"fourgrams_channels"`
	FivegramsChannels     int               `json:"fivegrams_channels"`
	Skip1BigramsChannels  int               `json:"skip1bigrams_channels"`
	Skip2BigramsChannels  int               `json:"skip2bigrams_channels"`
	Skip1TrigramsChannels int               `json:"skip1trigrams_channels"`
	Skip2TrigramsChannels int               `json:"skip2trigrams_channels"`
	OutputSize            int               `json:"output_size"`
	TypeVocabSize         int               `json:"type_vocab_size"`
	VocabSize             int               `json:"vocab_size"`
	Id2Label              map[string]string `json:"id2label"`
}

func LoadConfig(file string) (Config, error) {
	var config Config
	configFile, err := os.Open(file)
	if err != nil {
		return Config{}, err
	}
	defer configFile.Close()
	err = json.NewDecoder(configFile).Decode(&config)
	if err != nil {
		return Config{}, err
	}
	return config, nil
}

type Model struct {
	Config     Config
	Vocabulary *vocabulary.Vocabulary
	//Embeddings      *Embeddings
	Encoder *Encoder
	//AttentionNet    *AttentionNet
	//FeatureNet      *FeatureNet
	Classifier *linear.Model
}

type Processor struct {
	nn.BaseProcessor
	//Embeddings      *EmbeddingsProcessor
	Encoder *EncoderProcessor
	//AttentionNet    *AttentionNetProcessor
	//FeatureNet      *FeatureNetProcessor
	Classifier *ClassifierProcessor
}

func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		//Embeddings:      m.Embeddings.NewProc(g).(*EmbeddingsProcessor),
		Encoder: m.Encoder.NewProc(g).(*EncoderProcessor),
		//AttentionNet:    m.AttentionNet.NewProc(g).(*AttentionNetProcessor),
		//FeatureNet:      m.FeatureNet.NewProc(g).(*FeatureNetProcessor),
		Classifier: m.Classifier.NewProc(g).(*ClassifierProcessor),
	}
}

func (p *Processor) Forward(_ ...ag.Node) []ag.Node {
	panic("prado: method not implemented")
}
