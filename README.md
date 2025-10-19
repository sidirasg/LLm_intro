# Text Tokenization and Embeddings in Deep Learning

## Why Text Needs to Be Numbered

- Deep learning models are based on processing numbers
- Language models are built on mathematics, and math is based on numbers
- Optimization, learning algorithms, and probabilities all require numerical representations
- **Text must be represented as numbers for computational processing**

## What is a Token?

A **token** is a piece of text that can be represented as an integer.

### Examples of Tokenization:
- **Character-level**: The word "HELLO" represented as five numbers (a vector with one number for each character)
- **Subword-level**: "HELLO" split as "HE-LLO" → two numbers representing two subwords
- **Word-level**: The entire word "HELLO" as a single token (one number)

## Why Not Use Characters as Tokens?

Simple character mapping (e.g., a→1, b→2, ... z→26) has several problems:

### Problem 1: Unicode Complexity
- Many different unicode systems exist for letters across languages

### Problem 2: Statistical Inefficiency
- Ignores statistical regularities in language
- Misses patterns and relationships between character sequences

### Problem 3: Memory Limitations
- Requires significant additional memory
- Severely limits the context window size

## How LLMs Actually Work

**Important**: LLMs don't work directly with tokens - they work with **embeddings**!

### The Processing Pipeline:
```
Text → Token ID → Embedding → [LLM Processing] → Unembedding → Token → Text
```

- Text is converted into tokens
- Tokens must be converted into embeddings before the LLM can process them
- LLMs modify embeddings for classification and generation tasks
- **It's all about the embeddings**

## What Are Embeddings?

**Embeddings** are dense numeric representations of tokens.

### Advantages Over Simple Integers:
1. **Efficiency**: More text can be represented using fewer numbers
2. **Semantic Relations**: Relationships across tokens can be represented

### Characteristics of Real Embeddings:
1. **High-dimensional**: Often more than 1000 dimensions
2. **Not human-interpretable**: Abstract mathematical representations
3. **Dynamic**: Modified during model calculations (e.g., the "dog" vector can change based on context)

## Tokenization Challenges

Finding the optimal tokenization strategy is difficult:

### Trade-offs:
- **Fewer tokens**:
  - ✓ Less memory usage
  - ✓ Improved generalization
  - ✗ Less efficient and effective

- **More tokens**:
  - ✓ Better text compression
  - ✓ Can convey more information
  - ✗ Requires more training

### Key Considerations:
- Statistical dependencies change across languages
- Dependencies vary between databases within the same language
- Tokenization must be learned from text
- Different texts will create different tokenization schemes

## Key Definitions

- **Encoder**: A function that maps text into integers
- **Decoder**: The opposite of encoder - a function (lookup table) that maps integers back into text
- **Inverse Property**: `decoder(encoder(x)) = x`

## Summary

- Text must be transformed into numbers before LLM processing
- A chunk of text (token) can be a character, subword, or full word
- Embeddings are dense representations of tokens
- Both tokenization and embeddings are learned from data
- There are many valid ways to create these schemes
