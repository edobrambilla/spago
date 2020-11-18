package prado

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/encoding/hashing"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
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
			Size:       config.Size,
			DBPath:     config.WordsMapFilename,
			ReadOnly:   true,
			ForceNewDB: config.DeletePreEmbeddings,
		}),
		Projection: hashing.New(
			config.ProjectionSize,
			config.Size,
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

func (m *Embeddings) NewProc(g *ag.Graph) nn.Processor {
	return &EmbeddingsProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
		model:            m,
		wordsLayer:       m.Word.NewProc(g).(*embeddings.Processor),
		unknownEmbedding: g.NewWrap(m.Word.GetEmbedding(wordpiecetokenizer.DefaultUnknownToken)),
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
			encoded[i] = p.unknownEmbedding
		}
		if words[i] == wordpiecetokenizer.DefaultSequenceSeparator {
			sequenceIndex++
		}
	}
	return encoded
}

func (p *EmbeddingsProcessor) getWordEmbeddings(words []string) []ag.Node {
	return p.wordsLayer.Encode(words)
}

func (p *EmbeddingsProcessor) Forward(_ ...ag.Node) []ag.Node {
	panic("Prado: Forward() method not implemented. Use EmbedSequence() instead.")
}
