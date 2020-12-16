package prado

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/encoding/hashing"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
	"math"
)

var (
	_ nn.Model     = &Embeddings{}
	_ nn.Processor = &EmbeddingsProcessor{}
)

type EmbeddingsConfig struct {
	Size                int
	ProjectionSize      int
	ProjectionArity     int
	WordsMapFilename    string
	DeletePreEmbeddings bool
}

type Embeddings struct {
	EmbeddingsConfig
	Word       *embeddings.Model
	Projection *hashing.Data
}

func NewPradoEmbeddings(config EmbeddingsConfig) *Embeddings {
	return &Embeddings{
		EmbeddingsConfig: config,
		Word: embeddings.New(embeddings.Config{
			Size:             config.ProjectionSize, //embeddings into storage
			DBPath:           config.WordsMapFilename,
			ReadOnly:         false,
			ForceNewDB:       config.DeletePreEmbeddings,
			UseZeroEmbedding: false,
		}),
		Projection: hashing.New(
			config.Size,           //input size
			config.ProjectionSize, // output (projection) vectors size. These are the embeddings
			config.ProjectionArity),
	}
}

type EmbeddingsProcessor struct {
	nn.BaseProcessor
	model               *Embeddings
	wordsLayer          *embeddings.Processor
	tokenTypeEmbeddings []ag.Node
	unknownEmbedding    ag.Node
}

// Set Embeddings from a map of string into matrix. Codes may be calculated from string ascii, bytes or
// other type of embeddings
func (m *Embeddings) SetProjectedEmbeddings(codes map[string]mat.Matrix) {
	for word, code := range codes {
		hashedCode := m.Projection.GetHash(code)
		m.Word.SetEmbeddingFromData(word, hashedCode.Data())
	}
}

func (m *Embeddings) NewProc(ctx nn.Context) nn.Processor {
	graph := ctx.Graph
	return &EmbeddingsProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             graph,
			FullSeqProcessing: false,
		},
		model:            m,
		wordsLayer:       m.Word.NewProc(ctx).(*embeddings.Processor),
		unknownEmbedding: graph.NewWrap(graph.NewVariable(m.Projection.GetHash(mat.NewInitDense(m.Size, 1, -1.0)), false)),
	}
}

func (p *EmbeddingsProcessor) EmbedSequence(words []string) []ag.Node {
	encoded := make([]ag.Node, len(words))
	wordEmbeddings := p.getWordEmbeddings(words)
	sequenceIndex := 0
	for i := 0; i < len(words); i++ {
		if wordEmbeddings[i] != nil {
			encoded[i] = wordEmbeddings[i]
		} else {
			code := getStringCode(words[i])
			encoded[i] = p.Graph.NewVariable(p.model.Projection.GetHash(code), false)
		}
		if words[i] == wordpiecetokenizer.DefaultSequenceSeparator {
			sequenceIndex++
		}
	}
	return encoded
}

func getStringCode(s string) mat.Matrix {
	out := mat.NewEmptyVecDense(30)
	c := 0
	for _, char := range s {
		if c < 30 {
			for n := 1; n <= 3; n++ {
				out.Data()[c] = float64(digit(int(char), n))
				c++
			}
		}
	}
	return out.ProdScalar(0.1)
}

func digit(num, place int) int {
	r := num % int(math.Pow(10, float64(place)))
	return r / int(math.Pow(10, float64(place-1)))
}

func (p *EmbeddingsProcessor) getWordEmbeddings(words []string) []ag.Node {
	return p.wordsLayer.Encode(words)
}

func (p *EmbeddingsProcessor) Forward(_ ...ag.Node) []ag.Node {
	panic("Prado: Forward() method not implemented. Use EmbedSequence() instead.")
}
