// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package trainer

import (
	"bufio"
	"fmt"
	"github.com/gosuri/uiprogress"
	"github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/ml/losses"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/recurrent/lstm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/recurrent/srnn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
	"github.com/nlpodyssey/spago/pkg/utils"
	"io"
	"log"
	"os"
	"strconv"
)

type BiRNNTrainer struct {
	TrainingConfig
	randGen       *rand.LockedRand
	optimizer     *gd.GradientDescent
	bestLoss      float32
	lastBatchLoss float32
	model         *BiRNNClassifierModel
	countLines    int
	curLoss       float32
	curEpoch      int
	xDNNExamples  []*xDNNExample
}

type BiRNNClassifierConfig struct {
	EncodingActivation string            `json:"hidden_act"`
	InputSize          int               `json:"input_size"`
	OutputSize         int               `json:"output_size"`
	TypeVocabSize      int               `json:"type_vocab_size"`
	VocabSize          int               `json:"vocab_size"`
	Id2Label           map[string]string `json:"id2label"`
}

type BiRNNClassifierModel struct {
	nn.BaseModel
	Config     BiRNNClassifierConfig
	Embeddings *embeddings.Model
	//BiRNN  	*birnn.Model
	BiRNN      *srnn.BiModel
	Classifier *stack.Model
	Labels     []string
}

func NewBiSRNN(input, hidden int) *srnn.BiModel {
	config := srnn.Config{
		InputSize:  input,
		HiddenSize: 1024, //2048,
		NumLayers:  2,
		HyperSize:  32,
		OutputSize: hidden,
		MultiHead:  true,
	}
	return srnn.NewBidirectional(config)
}

func (m *BiRNNClassifierModel) InitBiRNNParameters(rndGen *rand.LockedRand) {
	initStacked(m.Classifier, rndGen)
	initStacked(m.BiRNN.FC, rndGen)
	initLinear(m.BiRNN.FC2, rndGen)
	initLinear(m.BiRNN.FC3, rndGen)
	initLayernorm(m.BiRNN.LayerNorm, rndGen)
	//initRecurrent(m.BiRNN.Negative.(*lstm.Model), rndGen)
	//initRecurrent(m.BiRNN.Positive.(*lstm.Model), rndGen)
}

func initStacked(model *stack.Model, rndGen *rand.LockedRand) {
	for i := 0; i < len(model.Layers)-1; i += 2 {
		layer := model.Layers[i]
		nextLayer := model.Layers[i+1]
		gain := 1.0
		if nextLayer, ok := nextLayer.(*activation.Model); ok {
			gain = float64(initializers.Gain(nextLayer.Activation))
		}
		nn.ForEachParam(layer, func(param nn.Param) {
			if param.Type() == nn.Weights {
				initializers.XavierUniform(param.Value(), mat32.Float(gain), rndGen)
			}
		})
	}
}

func initLinear(model *linear.Model, rndGen *rand.LockedRand) {
	gain := 1.0
	initializers.XavierUniform(model.W.Value(), mat32.Float(gain), rndGen)
}

func initLayernorm(model *layernorm.Model, rndGen *rand.LockedRand) {
	gain := 1.0
	initializers.XavierUniform(model.W.Value(), mat32.Float(gain), rndGen)
}

func initRecurrent(model *lstm.Model, rndGen *rand.LockedRand) {
	gain := 1.0
	initializers.XavierUniform(model.WCand.Value(), mat32.Float(gain), rndGen)
	initializers.XavierUniform(model.WCandRec.Value(), mat32.Float(gain), rndGen)
	initializers.XavierUniform(model.WFor.Value(), mat32.Float(gain), rndGen)
	initializers.XavierUniform(model.WForRec.Value(), mat32.Float(gain), rndGen)
	initializers.XavierUniform(model.WIn.Value(), mat32.Float(gain), rndGen)
	initializers.XavierUniform(model.WInRec.Value(), mat32.Float(gain), rndGen)
	initializers.XavierUniform(model.WOut.Value(), mat32.Float(gain), rndGen)
	initializers.XavierUniform(model.WOutRec.Value(), mat32.Float(gain), rndGen)
}

func NewBiRNNClassifierModel(config BiRNNClassifierConfig) *BiRNNClassifierModel {
	return &BiRNNClassifierModel{
		Config: config,
		Embeddings: embeddings.New(embeddings.Config{
			Size:             config.InputSize,
			UseZeroEmbedding: false,
			DBPath:           "path2",
			ReadOnly:         false,
			ForceNewDB:       true,
		}),
		//BiRNN: birnn.NewBiLSTM(config.InputSize, config.OutputSize, birnn.Concat),
		BiRNN: NewBiSRNN(config.InputSize, config.OutputSize),
		Classifier: stack.New(
			linear.New(2*config.OutputSize, len(config.Id2Label)),
			activation.New(ag.OpIdentity),
		),
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
	}
}

func NewBiRNNTrainer(model *BiRNNClassifierModel, config TrainingConfig, optimizer *gd.GradientDescent) *BiRNNTrainer {
	return &BiRNNTrainer{
		TrainingConfig: config,
		randGen:        rand.NewLockedRand(config.Seed),
		optimizer:      optimizer,
		model:          model,
		xDNNExamples:   make([]*xDNNExample, 0),
	}
}

func (m *BiRNNClassifierModel) EmbedSequence(words []string) []ag.Node {
	encoded := make([]ag.Node, len(words))
	wordEmbeddings := m.Embeddings.Encode(words)
	sequenceIndex := 0
	//r := rand.NewLockedRand(40)
	for i := 0; i < len(words); i++ {
		if wordEmbeddings[i] != nil {
			encoded[i] = wordEmbeddings[i]
		} else {
			encoded[i] = m.Embeddings.ZeroEmbedding
		}
		if words[i] == wordpiecetokenizer.DefaultSequenceSeparator {
			sequenceIndex++
		}
	}
	return encoded
}

func (m *BiRNNClassifierModel) EncodeText(tokens []string) ag.Node {
	e := m.EmbedSequence(tokens)
	encodedSequence := m.BiRNN.Forward(e...)
	textEncoding := m.Graph().Concat(encodedSequence[0], encodedSequence[len(tokens)-1])
	return textEncoding
}

func (m *BiRNNClassifierModel) Forward(tokens []string) []ag.Node {
	e := m.EncodeText(tokens)
	output := m.Classifier.Forward(e)
	return output
}

func (m *BiRNNClassifierModel) Classify(encodedText ag.Node) []ag.Node {
	output := m.Classifier.Forward(encodedText)
	return output
}

func (m *BiRNNClassifierModel) SetEmbeddings(v vocabulary.Vocabulary) {
	r := rand.NewLockedRand(40)
	for _, word := range v.Items() {
		code := mat32.NewEmptyVecDense(m.Config.InputSize)
		initializers.XavierUniform(code, initializers.Gain(ag.OpIdentity), r)
		m.Embeddings.SetEmbeddingFromData(word, code.Data())
	}
}

func (t *BiRNNTrainer) GetVocabulary() *vocabulary.Vocabulary {
	out := vocabulary.New([]string{})
	err := t.forEachLine(func(i int, text string) {
		e := GetExample(text)
		tokenizedExample := GetTokenizedExample(e, t.TrainingConfig.IncludeTitle, t.TrainingConfig.IncludeBody)
		tokenizedExample = PadTokens(tokenizedExample, 5)
		for _, word := range tokenizedExample {
			out.Add(word)
		}
		t.countLines++
	})
	if err != nil && err != io.EOF {
		fmt.Printf(" > Failed with error: %v\n", err)
	}
	return out
}

func (t *BiRNNTrainer) trainBatches(onExample func()) {
	err := t.forEachLine(func(i int, text string) {
		e := GetExample(text)
		tokenizedExample := GetTokenizedExample(e, t.TrainingConfig.IncludeTitle, t.TrainingConfig.IncludeBody)
		if len(tokenizedExample) > 0 {
			tokenizedExample = PadTokens(tokenizedExample, 5)
			t.curLoss = t.learn(i, tokenizedExample, t.TrainingConfig.LabelsMap[e.Category])
			t.optimizer.IncBatch()
			t.optimizer.IncExample()
			t.optimizer.Optimize()
		}
		onExample()
	})
	if err != nil && err != io.EOF {
		fmt.Printf(" > Failed with error: %v\n", err)
	}
}

// learn performs the backward respect to the cross-entropy loss, returned as scalar value
func (t *BiRNNTrainer) learn(_ int, tokenizedExample []string, label int) float32 {
	g := ag.NewGraph(ag.Rand(rand.NewLockedRand(t.Seed)))
	c := nn.Context{Graph: g, Mode: nn.Training}
	model := nn.Reify(c, t.model).(*BiRNNClassifierModel)
	defer g.Clear()

	encodedText := model.EncodeText(tokenizedExample)
	y := model.Classify(encodedText)[0]
	t.xDNNExamples = append(t.xDNNExamples, &xDNNExample{ //for xdnn classifier
		Category:      label,
		TokenizedText: tokenizedExample,
		BiRNNVector:   *encodedText.Value().(*mat32.Dense),
	})
	loss := g.Div(losses.CrossEntropy(g, y, label), g.NewScalar(1.0))
	g.Backward(loss)
	return loss.ScalarValue()
}

func (t *BiRNNTrainer) forEachLine(callback func(i int, line string)) (err error) {
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

func (t *BiRNNTrainer) newTrainBar(progress *uiprogress.Progress, nexamples int) *uiprogress.Bar {
	bar := progress.AddBar(nexamples)
	bar.AppendCompleted().PrependElapsed()
	bar.PrependFunc(func(b *uiprogress.Bar) string {
		return fmt.Sprintf("Epoch: %d Loss: %.6f", t.curEpoch, t.curLoss)
	})
	return bar
}

func (t *BiRNNTrainer) Enjoy() {
	for epoch := 1; epoch <= t.Epochs; epoch++ {
		t.curEpoch = epoch
		t.optimizer.IncEpoch()

		fmt.Println("Training epoch...")
		t.trainEpoch()

		// model serialization
		err := utils.SerializeToFile(t.ModelPath, t.model)
		if err != nil {
			panic("mnist: error during model serialization.")
		}
	}
}

func (t *BiRNNTrainer) trainEpoch() {
	uip := uiprogress.New()
	bar := t.newTrainBar(uip, t.countLines)
	uip.Start() // start bar rendering
	t.trainBatches(func() { bar.Incr() })
	uip.Stop()
	precision := NewEvaluator(t.model, t.TrainingConfig, "birnn").Evaluate(t.curEpoch).Precision()
	fmt.Printf("Accuracy: %.2f\n", 100*precision)
}
