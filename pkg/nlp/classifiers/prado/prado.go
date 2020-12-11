// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package prado

import (
	"encoding/json"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/convolution"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
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
	EncodingActivation    string            `json:"hidden_act"`
	ConvActivation        string            `json:"conv_act"`
	ConvSize              int               `json:"conv_size"`
	InputSize             int               `json:"input_size"`
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

func mustGetOpName(str string) ag.OpName {
	if value, err := ag.GetOpName(str); err == nil {
		return value
	} else {
		panic(err)
	}
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
	Config       Config
	Vocabulary   *vocabulary.Vocabulary
	Embeddings   *Embeddings
	Encoder      *Encoder
	AttentionNet *FeatureNet
	FeatureNet   *FeatureNet
	TextEncoder  *TextEncoder
	Classifier   *Classifier
}

func NewDefaultPrado(config Config, embeddingsStoragePath string) *Model {
	nChannels := config.UnigramsChannels + config.BigramsChannels + config.TrigramsChannels + config.FourgramsChannels +
		config.FivegramsChannels + config.Skip1BigramsChannels + config.Skip1TrigramsChannels +
		config.Skip2BigramsChannels
	return &Model{
		Config:     config,
		Vocabulary: nil,
		Embeddings: NewPradoEmbeddings(EmbeddingsConfig{
			Size:                config.InputSize,
			ProjectionSize:      config.ProjectionSize,
			ProjectionArity:     3,
			WordsMapFilename:    embeddingsStoragePath,
			DeletePreEmbeddings: true,
		}),
		Encoder: NewPradoEncoder(EncoderConfig{
			InputSize:   config.ProjectionSize,
			EncodedSize: config.EncodingSize,
			Activation:  mustGetOpName(config.EncodingActivation),
		}),
		AttentionNet: NewFeatureNet(FeatureNetConfig{
			EncodingSize:          config.EncodingSize,
			UnigramsChannels:      config.UnigramsChannels,
			BigramsChannels:       config.BigramsChannels,
			TrigramsChannels:      config.TrigramsChannels,
			FourgramsChannels:     config.FourgramsChannels,
			FivegramsChannels:     config.FivegramsChannels,
			Skip1BigramsChannels:  config.Skip1BigramsChannels,
			Skip2BigramsChannels:  config.Skip2BigramsChannels,
			Skip1TrigramsChannels: config.Skip1TrigramsChannels,
			AttentionNet:          true,
			ConvSize:              config.ConvSize,
			ConvActivation:        config.ConvActivation,
		}),
		FeatureNet: NewFeatureNet(FeatureNetConfig{
			EncodingSize:          config.EncodingSize,
			UnigramsChannels:      config.UnigramsChannels,
			BigramsChannels:       config.BigramsChannels,
			TrigramsChannels:      config.TrigramsChannels,
			FourgramsChannels:     config.FourgramsChannels,
			FivegramsChannels:     config.FivegramsChannels,
			Skip1BigramsChannels:  config.Skip1BigramsChannels,
			Skip2BigramsChannels:  config.Skip2BigramsChannels,
			Skip1TrigramsChannels: config.Skip1TrigramsChannels,
			AttentionNet:          false,
			ConvSize:              config.ConvSize,
			ConvActivation:        config.ConvActivation,
		}),
		TextEncoder: NewPradoTextEncoder(),
		Classifier: NewPradoClassifier(ClassifierConfig{
			TextEncodingSize: nChannels * (config.EncodingSize - config.ConvSize + 1),
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

func (m *Model) InitPradoParameters(rndGen *rand.LockedRand) {
	initStacked(m.Encoder.Model, rndGen)
	initStacked(m.Classifier.Model, rndGen)
	initConvolution(m.FeatureNet.convolutionModels, rndGen)
	initConvolution(m.AttentionNet.convolutionModels, rndGen)
}

// InitRandom initializes the model using the Xavier (Glorot) method.
func initStacked(model *stack.Model, rndGen *rand.LockedRand) {
	for i := 0; i < len(model.Layers)-1; i += 2 {
		layer := model.Layers[i]
		nextLayer := model.Layers[i+1]
		gain := 1.0
		if nextLayer, ok := nextLayer.(*activation.Model); ok {
			gain = initializers.Gain(nextLayer.Activation)
		}
		nn.ForEachParam(layer, func(param *nn.Param) {
			if param.Type() == nn.Weights {
				initializers.XavierUniform(param.Value(), gain, rndGen)
			}
		})
	}
}

// InitCNN initializes the model using the Xavier (Glorot) method.
func initConvolution(models []*convolution.Model, rndGen *rand.LockedRand) {
	for c := 0; c < len(models); c++ {
		for i := 0; i < len(models[c].K); i++ {

			initializers.XavierUniform(models[c].K[i].Value(), initializers.Gain(models[c].Activation), rndGen)
			initializers.XavierUniform(models[c].B[i].Value(), initializers.Gain(models[c].Activation), rndGen)
		}
	}

}

type Processor struct {
	nn.BaseProcessor
	Embeddings   *EmbeddingsProcessor
	Encoder      *EncoderProcessor
	AttentionNet *FeatureNetProcessor
	FeatureNet   *FeatureNetProcessor
	TextEncoder  *TextEncoderProcessor
	Classifier   *ClassifierProcessor
}

func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		Embeddings:   m.Embeddings.NewProc(g).(*EmbeddingsProcessor),
		Encoder:      m.Encoder.NewProc(g).(*EncoderProcessor),
		AttentionNet: m.FeatureNet.NewProc(g).(*FeatureNetProcessor),
		FeatureNet:   m.FeatureNet.NewProc(g).(*FeatureNetProcessor),
		TextEncoder:  m.TextEncoder.NewProc(g).(*TextEncoderProcessor),
		Classifier:   m.Classifier.NewProc(g).(*ClassifierProcessor),
	}
}

func (p *Processor) Classify(tokens []string) []ag.Node {
	e := p.Embeddings.EmbedSequence(tokens)
	encodedSequence := p.Encoder.Encode(e)
	featureNetEncoding := p.FeatureNet.Encode(p.Model.(*Model).FeatureNet.config, encodedSequence...)
	attentionNetEncoding := p.AttentionNet.Encode(p.Model.(*Model).AttentionNet.config, encodedSequence...)
	textEncoding := p.TextEncoder.Encode(featureNetEncoding, attentionNetEncoding)
	output := p.Classifier.Forward(textEncoding)
	return output
}

func (p *Processor) Forward(_ ...ag.Node) []ag.Node {
	panic("prado: method not implemented")
}
