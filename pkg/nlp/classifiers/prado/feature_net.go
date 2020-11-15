// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package prado

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/convolution"
	"github.com/nlpodyssey/spago/pkg/utils/data"
)

var (
	_ nn.Model     = &FeatureNet{}
	_ nn.Processor = &FeatureNetProcessor{}
)

type FeatureNetConfig struct {
	EncodingSize          int
	UnigramsChannels      int
	BigramsChannels       int
	TrigramsChannels      int
	FourgramsChannels     int
	FivegramsChannels     int
	Skip1BigramsChannels  int
	Skip2BigramsChannels  int
	Skip1TrigramsChannels int
	AttentionNet          bool
	OutputSize            int
}

type FeatureNet struct {
	config            FeatureNetConfig
	convolutionModels []*convolution.Model
}

func New(config FeatureNetConfig) *FeatureNet {
	nChannels := config.UnigramsChannels + config.BigramsChannels + config.TrigramsChannels + config.FourgramsChannels +
		config.FivegramsChannels + config.Skip1BigramsChannels + config.Skip1TrigramsChannels +
		config.Skip2BigramsChannels
	convolutionModels := make([]*convolution.Model, nChannels)
	c := 0
	for n := 0; n < config.UnigramsChannels; n++ {
		convolutionModels[c+n] = convolution.New(convolution.Config{
			KernelSizeX:    4, //todo calculate output
			KernelSizeY:    1,
			XStride:        0,
			YStride:        0,
			InputChannels:  1,
			OutputChannels: 1, //todo calculate output
			Mask:           nil,
			Activation:     0,
		})
	}
	c += config.UnigramsChannels
	for n := 0; n < config.BigramsChannels; n++ {
		convolutionModels[c+n] = convolution.New(convolution.Config{
			KernelSizeX:    4, //todo calculate output
			KernelSizeY:    1,
			XStride:        0,
			YStride:        0,
			InputChannels:  2,
			OutputChannels: 1, //todo calculate output
			Mask:           nil,
			Activation:     0,
		})
	}
	c += config.BigramsChannels
	for n := 0; n < config.TrigramsChannels; n++ {
		convolutionModels[c+n] = convolution.New(convolution.Config{
			KernelSizeX:    4, //todo calculate output
			KernelSizeY:    1,
			XStride:        0,
			YStride:        0,
			InputChannels:  3,
			OutputChannels: 1, //todo calculate output
			Mask:           nil,
			Activation:     0,
		})
	}
	c += config.TrigramsChannels
	for n := 0; n < config.FourgramsChannels; n++ {
		convolutionModels[c+n] = convolution.New(convolution.Config{
			KernelSizeX:    4, //todo calculate output
			KernelSizeY:    1,
			XStride:        0,
			YStride:        0,
			InputChannels:  4,
			OutputChannels: 1, //todo calculate output
			Mask:           nil,
			Activation:     0,
		})
	}
	c += config.FourgramsChannels
	for n := 0; n < config.FivegramsChannels; n++ {
		convolutionModels[c+n] = convolution.New(convolution.Config{
			KernelSizeX:    4, //todo calculate output
			KernelSizeY:    1,
			XStride:        0,
			YStride:        0,
			InputChannels:  4,
			OutputChannels: 1, //todo calculate output
			Mask:           nil,
			Activation:     0,
		})
	}
	c += config.FivegramsChannels
	for n := 0; n < config.Skip1BigramsChannels; n++ {
		convolutionModels[c+n] = convolution.New(convolution.Config{
			KernelSizeX:    4, //todo calculate output
			KernelSizeY:    1,
			XStride:        0,
			YStride:        0,
			InputChannels:  3,
			OutputChannels: 1, //todo calculate output
			Mask:           []int{1, 0, 1},
			Activation:     0,
		})
	}
	c += config.Skip1BigramsChannels
	for n := 0; n < config.Skip2BigramsChannels; n++ {
		convolutionModels[c+n] = convolution.New(convolution.Config{
			KernelSizeX:    4, //todo calculate output
			KernelSizeY:    1,
			XStride:        0,
			YStride:        0,
			InputChannels:  4,
			OutputChannels: 1, //todo calculate output
			Mask:           []int{1, 0, 0, 1},
			Activation:     0,
		})
	}
	c += config.Skip2BigramsChannels
	for n := 0; n < config.Skip1TrigramsChannels; n++ {
		convolutionModels[c+n] = convolution.New(convolution.Config{
			KernelSizeX:    4, //todo calculate output
			KernelSizeY:    1,
			XStride:        0,
			YStride:        0,
			InputChannels:  4,
			OutputChannels: 1, //todo calculate output
			Mask:           []int{1, 1, 0, 1},
			Activation:     0,
		})
	}
	return &FeatureNet{
		config:            config,
		convolutionModels: convolutionModels,
	}
}

type FeatureNetProcessor struct {
	nn.BaseProcessor
	convolutionProcessors []*convolution.Processor
}

func (m *FeatureNet) NewProc(g *ag.Graph) nn.Processor {
	nChannels := m.config.UnigramsChannels + m.config.BigramsChannels + m.config.TrigramsChannels + m.config.FourgramsChannels +
		m.config.FivegramsChannels + m.config.Skip1BigramsChannels + m.config.Skip1TrigramsChannels +
		m.config.Skip2BigramsChannels
	convNetProc := make([]*convolution.Processor, nChannels)
	c := 0
	for n := 0; n < nChannels; n++ {
		convNetProc[n] = m.convolutionModels[c+n].NewProc(g).(*convolution.Processor)
	}
	return &FeatureNetProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		convolutionProcessors: convNetProc,
	}
}

func (p *FeatureNetProcessor) Encode(config FeatureNetConfig, xs ...ag.Node) [][]ag.Node {

	nChannels := config.UnigramsChannels + config.BigramsChannels + config.TrigramsChannels + config.FourgramsChannels +
		config.FivegramsChannels + config.Skip1BigramsChannels + config.Skip1TrigramsChannels +
		config.Skip2BigramsChannels
	out := make([][]ag.Node, nChannels)
	unigrams := data.GenerateNGrams(1, len(xs))
	bigrams := data.GenerateNGrams(2, len(xs))
	trigrams := data.GenerateNGrams(3, len(xs))
	fourgrams := data.GenerateNGrams(4, len(xs))
	fivgrams := data.GenerateNGrams(5, len(xs))
	c := 0
	for n := 0; n < config.UnigramsChannels; n++ {
		fn := make([]ag.Node, len(unigrams))
		for i, unigram := range unigrams {
			fn[i] = p.convolutionProcessors[c+n].Forward(xs[unigram[0]])[0]
		}
		out[c+n] = fn
	}
	c += config.UnigramsChannels
	for n := 0; n < config.BigramsChannels; n++ {
		fn := make([]ag.Node, len(bigrams))
		for i, bigram := range bigrams {
			fn[i] = p.convolutionProcessors[c+n].Forward(xs[bigram[0]], xs[bigram[1]])[0]
		}
		out[c+n] = fn
	}
	c += config.BigramsChannels
	for n := 0; n < config.TrigramsChannels; n++ {
		fn := make([]ag.Node, len(trigrams))
		for i, trigram := range bigrams {
			fn[i] = p.convolutionProcessors[c+n].Forward(xs[trigram[0]], xs[trigram[1]], xs[trigram[2]])[0]
		}
		out[c+n] = fn
	}
	c += config.TrigramsChannels
	for n := 0; n < config.FourgramsChannels; n++ {
		fn := make([]ag.Node, len(fourgrams))
		for i, fourgram := range bigrams {
			fn[i] = p.convolutionProcessors[c+n].Forward(xs[fourgram[0]], xs[fourgram[1]], xs[fourgram[2]],
				xs[fourgram[3]])[0]
		}
		out[c+n] = fn
	}
	c += config.FourgramsChannels
	for n := 0; n < config.FivegramsChannels; n++ {
		fn := make([]ag.Node, len(fivgrams))
		for i, fivegram := range bigrams {
			fn[i] = p.convolutionProcessors[c+n].Forward(xs[fivegram[0]], xs[fivegram[1]], xs[fivegram[2]],
				xs[fivegram[3]], xs[fivegram[4]])[0]
		}
		out[c+n] = fn
	}
	c += config.FivegramsChannels
	for n := 0; n < config.Skip1BigramsChannels; n++ {
		fn := make([]ag.Node, len(trigrams))
		for i, trigram := range bigrams {
			fn[i] = p.convolutionProcessors[c+n].Forward(xs[trigram[0]], xs[trigram[1]], xs[trigram[2]])[0]
		}
		out[c+n] = fn
	}
	c += config.Skip1BigramsChannels
	for n := 0; n < config.Skip2BigramsChannels; n++ {
		fn := make([]ag.Node, len(fourgrams))
		for i, fourgram := range bigrams {
			fn[i] = p.convolutionProcessors[c+n].Forward(xs[fourgram[0]], xs[fourgram[1]], xs[fourgram[2]],
				xs[fourgram[3]])[0]
		}
		out[c+n] = fn
	}
	c += config.Skip2BigramsChannels
	for n := 0; n < config.Skip1TrigramsChannels; n++ {
		fn := make([]ag.Node, len(fourgrams))
		for i, fourgram := range bigrams {
			fn[i] = p.convolutionProcessors[c+n].Forward(xs[fourgram[0]], xs[fourgram[1]], xs[fourgram[2]],
				xs[fourgram[3]])[0]
		}
		out[c+n] = fn
	}
	return out
}

func (p *FeatureNetProcessor) Forward(_ ...ag.Node) []ag.Node {
	panic("Prado: Forward() method not implemented. Use Encode() instead.")
}
