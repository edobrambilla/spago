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
	_ nn.Model = &FeatureNet{}
)

type FeatureNetConfig struct {
	EncodingSize          int
	ConvSize              int
	ConvActivation        string
	UnigramsChannels      int
	BigramsChannels       int
	TrigramsChannels      int
	FourgramsChannels     int
	FivegramsChannels     int
	Skip1BigramsChannels  int
	Skip2BigramsChannels  int
	Skip1TrigramsChannels int
	AttentionNet          bool
}

type FeatureNet struct {
	nn.BaseModel
	Config            FeatureNetConfig
	ConvolutionModels []*convolution.Model
}

func NewFeatureNet(config FeatureNetConfig) *FeatureNet {
	nChannels := config.UnigramsChannels + config.BigramsChannels + config.TrigramsChannels + config.FourgramsChannels +
		config.FivegramsChannels + config.Skip1BigramsChannels + config.Skip1TrigramsChannels +
		config.Skip2BigramsChannels
	convolutionModels := make([]*convolution.Model, nChannels)
	c := 0
	for n := 0; n < config.UnigramsChannels; n++ {
		convolutionModels[c+n] = convolution.New(convolution.Config{
			KernelSizeX:    config.ConvSize,
			KernelSizeY:    1,
			XStride:        1,
			YStride:        1,
			InputChannels:  1,
			OutputChannels: 1,
			Mask:           nil,
			Activation:     mustGetOpName(config.ConvActivation),
		})
	}
	c += config.UnigramsChannels
	for n := 0; n < config.BigramsChannels; n++ {
		convolutionModels[c+n] = convolution.New(convolution.Config{
			KernelSizeX:    config.ConvSize,
			KernelSizeY:    1,
			XStride:        1,
			YStride:        1,
			InputChannels:  2,
			OutputChannels: 1,
			Mask:           nil,
			Activation:     mustGetOpName(config.ConvActivation),
		})
	}
	c += config.BigramsChannels
	for n := 0; n < config.TrigramsChannels; n++ {
		convolutionModels[c+n] = convolution.New(convolution.Config{
			KernelSizeX:    config.ConvSize,
			KernelSizeY:    1,
			XStride:        1,
			YStride:        1,
			InputChannels:  3,
			OutputChannels: 1,
			Mask:           nil,
			Activation:     mustGetOpName(config.ConvActivation),
		})
	}
	c += config.TrigramsChannels
	for n := 0; n < config.FourgramsChannels; n++ {
		convolutionModels[c+n] = convolution.New(convolution.Config{
			KernelSizeX:    config.ConvSize,
			KernelSizeY:    1,
			XStride:        1,
			YStride:        1,
			InputChannels:  4,
			OutputChannels: 1,
			Mask:           nil,
			Activation:     mustGetOpName(config.ConvActivation),
		})
	}
	c += config.FourgramsChannels
	for n := 0; n < config.FivegramsChannels; n++ {
		convolutionModels[c+n] = convolution.New(convolution.Config{
			KernelSizeX:    config.ConvSize,
			KernelSizeY:    1,
			XStride:        1,
			YStride:        1,
			InputChannels:  5,
			OutputChannels: 1,
			Mask:           nil,
			Activation:     mustGetOpName(config.ConvActivation),
		})
	}
	c += config.FivegramsChannels
	for n := 0; n < config.Skip1BigramsChannels; n++ {
		convolutionModels[c+n] = convolution.New(convolution.Config{
			KernelSizeX:    config.ConvSize,
			KernelSizeY:    1,
			XStride:        1,
			YStride:        1,
			InputChannels:  3,
			OutputChannels: 1,
			Mask:           []int{1, 0, 1},
			Activation:     mustGetOpName(config.ConvActivation),
		})
	}
	c += config.Skip1BigramsChannels
	for n := 0; n < config.Skip2BigramsChannels; n++ {
		convolutionModels[c+n] = convolution.New(convolution.Config{
			KernelSizeX:    config.ConvSize,
			KernelSizeY:    1,
			XStride:        1,
			YStride:        1,
			InputChannels:  4,
			OutputChannels: 1,
			Mask:           []int{1, 0, 0, 1},
			Activation:     mustGetOpName(config.ConvActivation),
		})
	}
	c += config.Skip2BigramsChannels
	for n := 0; n < config.Skip1TrigramsChannels; n++ {
		convolutionModels[c+n] = convolution.New(convolution.Config{
			KernelSizeX:    config.ConvSize,
			KernelSizeY:    1,
			XStride:        1,
			YStride:        1,
			InputChannels:  4,
			OutputChannels: 1,
			Mask:           []int{1, 1, 0, 1},
			Activation:     mustGetOpName(config.ConvActivation),
		})
	}
	return &FeatureNet{
		Config:            config,
		ConvolutionModels: convolutionModels,
	}
}

func (p *FeatureNet) calculateAttention(xs []ag.Node) []ag.Node {
	attention := make([]ag.Node, len(xs))
	sum := p.Graph().Exp(xs[0])
	for i := 1; i < len(xs); i++ {
		sum = p.Graph().Add(sum, p.Graph().Exp(xs[i]))
	}
	for i := 0; i < len(xs); i++ {
		attention[i] = p.Graph().Div(p.Graph().Exp(xs[i]), sum)
	}
	return attention
}

func (p *FeatureNet) encodeNgrams(a [][]int, c, ngramSize, channels int, attentioNet bool, out [][]ag.Node, xs ...ag.Node) {
	for n := 0; n < channels; n++ {
		fn := make([]ag.Node, len(a))
		for i, ngram := range a {
			switch ngramSize {
			case 1:
				fn[i] = p.ConvolutionModels[c+n].Forward(xs[ngram[0]])[0]
			case 2:
				fn[i] = p.ConvolutionModels[c+n].Forward(xs[ngram[0]], xs[ngram[1]])[0]
			case 3:
				fn[i] = p.ConvolutionModels[c+n].Forward(xs[ngram[0]], xs[ngram[1]], xs[ngram[2]])[0]
			case 4:
				fn[i] = p.ConvolutionModels[c+n].Forward(xs[ngram[0]], xs[ngram[1]], xs[ngram[2]], xs[ngram[3]])[0]
			case 5:
				fn[i] = p.ConvolutionModels[c+n].Forward(xs[ngram[0]], xs[ngram[1]], xs[ngram[2]], xs[ngram[3]], xs[ngram[4]])[0]
			}

		}
		if attentioNet {
			fn = p.calculateAttention(fn)
		}
		out[c+n] = fn
	}
}

func (p *FeatureNet) Encode(config FeatureNetConfig, xs ...ag.Node) [][]ag.Node {

	nChannels := p.Config.UnigramsChannels + p.Config.BigramsChannels + p.Config.TrigramsChannels + p.Config.FourgramsChannels +
		p.Config.FivegramsChannels + p.Config.Skip1BigramsChannels + p.Config.Skip1TrigramsChannels +
		p.Config.Skip2BigramsChannels
	out := make([][]ag.Node, nChannels)
	unigrams := data.GenerateNGrams(1, len(xs))
	bigrams := data.GenerateNGrams(2, len(xs))
	trigrams := data.GenerateNGrams(3, len(xs))
	fourgrams := data.GenerateNGrams(4, len(xs))
	fivgrams := data.GenerateNGrams(5, len(xs))
	c := 0
	p.encodeNgrams(unigrams, c, 1, p.Config.UnigramsChannels, config.AttentionNet, out, xs...)
	c += config.UnigramsChannels
	p.encodeNgrams(bigrams, c, 2, p.Config.BigramsChannels, config.AttentionNet, out, xs...)
	c += config.BigramsChannels
	p.encodeNgrams(trigrams, c, 3, p.Config.TrigramsChannels, config.AttentionNet, out, xs...)
	c += config.TrigramsChannels
	p.encodeNgrams(fourgrams, c, 4, p.Config.FourgramsChannels, config.AttentionNet, out, xs...)
	c += config.FourgramsChannels
	p.encodeNgrams(fivgrams, c, 5, p.Config.FivegramsChannels, config.AttentionNet, out, xs...)
	c += config.FivegramsChannels
	p.encodeNgrams(trigrams, c, 3, p.Config.Skip1BigramsChannels, config.AttentionNet, out, xs...)
	c += config.Skip1BigramsChannels
	p.encodeNgrams(fourgrams, c, 4, p.Config.Skip2BigramsChannels, config.AttentionNet, out, xs...)
	c += config.Skip2BigramsChannels
	p.encodeNgrams(fourgrams, c, 4, p.Config.Skip1TrigramsChannels, config.AttentionNet, out, xs...)
	return out
}
