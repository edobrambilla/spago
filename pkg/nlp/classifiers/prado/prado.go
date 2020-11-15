// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package prado

import (
	"encoding/json"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
	"log"
	"os"
	"strconv"
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
	ConvSize              int               `json:"conv_size"`
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
	Embeddings *Embeddings
	Encoder    *Encoder
	//AttentionNet   *AttentionNet
	FeatureNet  *FeatureNet
	TextEncoder *TextEncoder
	Classifier  *Classifier
}

// NewDefaultBERT returns a new model based on the original BERT architecture.
func NewDefaultBERT(config Config, embeddingsStoragePath string) *Model {
	nChannels := config.UnigramsChannels + config.BigramsChannels + config.TrigramsChannels + config.FourgramsChannels +
		config.FivegramsChannels + config.Skip1BigramsChannels + config.Skip1TrigramsChannels +
		config.Skip2BigramsChannels
	return &Model{
		Config:     config,
		Vocabulary: nil,
		Embeddings: &Embeddings{
			EmbeddingsConfig: EmbeddingsConfig{
				Size:                0,
				ProjectionSize:      config.ProjectionSize,
				ProjectionArity:     3,
				WordsMapFilename:    embeddingsStoragePath,
				DeletePreEmbeddings: false,
			},
			Word: &embeddings.Model{
				Config: embeddings.Config{
					Size:             0,
					UseZeroEmbedding: false,
					DBPath:           embeddingsStoragePath,
					ReadOnly:         false,
					ForceNewDB:       false,
				},
				UsedEmbeddings: nil,
				ZeroEmbedding:  &nn.Param{},
			},
		},
		Encoder: &Encoder{
			EncoderConfig: EncoderConfig{
				InputSize:   config.ProjectionSize,
				EncodedSize: config.EncodingSize,
				Activation:  ag.OpIdentity,
			},
		},
		TextEncoder: &TextEncoder{},
		Classifier: NewPradoClassifier(ClassifierConfig{
			TextEncodingSize: nChannels * config.ConvSize,
			Labels: func(x map[string]string) []string {
				if len(x) == 0 {
					return []string{"LABEL_0", "LABEL_1"} // assume binary classification by default
				}
				y := make([]string, len(x))
				for k, v := range x {
					i, err := strconv.Atoi(k)
					if err != nil {
						log.Fatal(err)
					}
					y[i] = v
				}
				return y
			}(config.Id2Label),
			Activation: ag.OpIdentity,
		}),
	}
}

type Processor struct {
	nn.BaseProcessor
	Embeddings *EmbeddingsProcessor
	Encoder    *EncoderProcessor
	//AttentionNet    *AttentionNetProcessor
	FeatureNet  *FeatureNetProcessor
	TextEncoder *TextEncoderProcessor
	Classifier  *ClassifierProcessor
}

func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		Embeddings: m.Embeddings.NewProc(g).(*EmbeddingsProcessor),
		Encoder:    m.Encoder.NewProc(g).(*EncoderProcessor),
		//AttentionNet:    m.AttentionNet.NewProc(g).(*AttentionNetProcessor),
		FeatureNet:  m.FeatureNet.NewProc(g).(*FeatureNetProcessor),
		TextEncoder: m.TextEncoder.NewProc(g).(*TextEncoderProcessor),
		Classifier:  m.Classifier.NewProc(g).(*ClassifierProcessor),
	}
}

func (p *Processor) Forward(_ ...ag.Node) []ag.Node {
	panic("prado: method not implemented")
}
