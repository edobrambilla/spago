package prado

import (
	"github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/encoding/hashing"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
	"math"
)

var (
	_ nn.Model = &Embeddings{}
)

type EmbeddingsConfig struct {
	InputSize           int
	ProjectionSize      int
	ProjectionArity     int
	WordsMapFilename    string
	DeletePreEmbeddings bool
}

type Embeddings struct {
	nn.BaseModel
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
			config.InputSize,      //input size
			config.ProjectionSize, // output (projection) vectors size. These are the embeddings
			config.ProjectionArity),
	}
}

// Set Embeddings from a map of string into matrix. Codes may be calculated from string ascii, bytes or
// other type of embeddings
func (m *Embeddings) SetProjectedEmbeddings(codes map[string]mat32.Matrix) {
	for word, code := range codes {
		hashedCode := m.Projection.GetHashProjection(code)
		m.Word.SetEmbeddingFromData(word, hashedCode.Data())
	}
}

func (p *Embeddings) EmbedSequence(words []string) []ag.Node {
	encoded := make([]ag.Node, len(words))
	wordEmbeddings := p.getWordEmbeddings(words)
	sequenceIndex := 0
	for i := 0; i < len(words); i++ {
		if wordEmbeddings[i] != nil {
			encoded[i] = wordEmbeddings[i]
		} else {
			//code := getStringCode(words[i], p.EmbeddingsConfig) // alternative way to calculate unknown embedding.
			encoded[i] = p.Word.ZeroEmbedding
		}
		if words[i] == wordpiecetokenizer.DefaultSequenceSeparator {
			sequenceIndex++
		}
	}
	return encoded
}

// Get code vector, binary random. Then this vector is transformed in nternary vector like original prado paper
// The code size must be N*2, where N is the projection size
func GetHashCode(config EmbeddingsConfig, r *rand.LockedRand) mat32.Matrix {
	out := mat32.NewEmptyVecDense(config.InputSize)
	c := 0
	for i := 0; i < config.InputSize; i++ {
		rnd := (r.Float32() * 2.0) - 1.0
		if rnd >= 0.0 {
			out.Data()[c] = 1.0
		} else {
			out.Data()[c] = 0.0
		}
		c++
	}
	return out
}

// Get code vector, string based: similar words get similar code. Then each vector is projected into ternary space
func GetStringCode(s string, config EmbeddingsConfig) mat32.Matrix {
	out := mat32.NewEmptyVecDense(config.InputSize)
	c := 0
	for _, char := range s {
		if c < config.InputSize-3 {
			for n := 1; n <= 3; n++ {
				f := -653
				if char%2 == 0 {
					f = 653
				}
				out.Data()[c] = mat32.Float(float64(digit(int(char*int32(f)), n)))
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

func (p *Embeddings) getWordEmbeddings(words []string) []ag.Node {
	return p.Word.Encode(words)
}
