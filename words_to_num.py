#why textr need to be numbered?
#because the models od Deep learninig based on porocessimng numbers
#Yes because Language models are based on math and manths is based on numbers
#OPtimaziation Learininig algorithms and probalities are based on numbers
#Text must be represented as numbeewrs
#Token A pice of text that can represened as an integer
#eg word HELLO rerpesented as five number (vector) a number for each character
#the word HGello can represented as subword we can split the HE-LLO and we will have tw numbers for this word these number is a subword
#ALternative we can have a token (a number) foir this word

#why not just use a caractedrs as a token eg
#1--->a
#2--->b
#26-->z

#problem first there are a lot of diferent unicode systems for  the leters
#problem 2 ignores statistical regularities in language
#problem3 Requares a lot of adiotional memory which limits the context window


#So LLMs work with tokens ? No
#Text is converted  into tokens but LLm work with embeddings
#token must be converted into embedding before the LLM
#LLms modify the embedding for the calssification and generation

#It's is all about the ebeddings

#text ---> TokenID---->Embedding--->unEmbedding ---->Token


#what is ebdeddings?
#A dense numeric representation of token

#Advandages over integer
#1 more text can represented ussing fewer numbers
#2 Semanic relation acros tokoens can represented

#REal emeding are : 1 Hight dimensional (maybe bore 1000 dimensions) 2 not human interepretable 3 modiofy dynamicaly during model calculation (eg the dog vector can change if the dog is separd)

#tokenization is dificalt
#fewer token means less memory and impoved generalazation but is less efficient and effective
#More tokens means more traininig but can increase text compression and convey more infoirmationg
#statistical dependencies change across languages and databases within a language.

#tokenization must be learned from text and diofferent text will create diferent tokenazations

#text must be transformed ing number before LLM
#A chuck of text is a token and can be a character, subword or full word
#emebdinng are dense representations of tokens
#tokenization and embending ate lerened from data amd there are many ways to create these schemes

#Encoder a function that maps text into integers
# Decodet is the opposite of encoder . A function (look up table) that maps integers into text\

# Encoders and decoderts are inverses: decoder(encoder(x)=x










#example of code
#1. split a sentence into lower case word
#2  create a vocab from unique word  (tokens)
#3  Create encoder and decoder function as dictinaris
#4 create a tokenization scheme
# see the result
# Example of tokenization code
# 1. Split a sentence into lower case words
# 2. Create a vocab from unique words (tokens)
# 3. Create encoder and decoder functions as dictionaries
# 4. Create a tokenization scheme
# 5. See the results

import re
import numpy as np
import matplotlib.pyplot as plt

# Sample text data
text = ["All that we are is the result of what we thought",
        "To be or not to be that is the question",
        "Be yourself everyone else is already taken"]

print("Original texts:")
for i, t in enumerate(text):
    print(f"  {i + 1}. {t}")
print()

# Separate into words by splitting by spaces
print("Example of splitting first sentence:")
print(re.split('\s', text[0]))

# Can recombine into text
print("\nRecombined text:")
print(' '.join(re.split('\s', text[0])))

# Also make lower-case and split all texts
allwords = re.split('\s', ' '.join(text).lower())
print("\nAll words (lower case):")
print(allwords)
print()

# ============================================
# Create a vocabulary (lexicon)
# ============================================

# Find the unique words
vocab = sorted(set(allwords))
print(f"Vocabulary ({len(vocab)} unique words):")
print(vocab)
print()

# ============================================
# Create an encoder and decoder
# ============================================

# The encoder is a python dictionary type
word2idx = {}
for i, word in enumerate(vocab):
    word2idx[word] = i

print("Word to Index mapping (first 5):")
for word, idx in list(word2idx.items())[:5]:
    print(f"  '{word}' -> {idx}")

# And a decoder
idx2word = {}
for i, word in enumerate(vocab):
    idx2word[i] = word

print("\nIndex to Word mapping (first 5):")
for idx, word in list(idx2word.items())[:5]:
    print(f"  {idx} -> '{word}'")

print(f'\nExample lookups:')
print(f'  The word "to" has index {word2idx["to"]}')
print(f'  The index "7" maps to the word "{idx2word[7]}"')
print()

# ============================================
# Make fake quotes, just for fun :P
# ============================================

# Select random words from the dictionary
np.random.seed(42)  # For reproducibility
randidx = np.random.randint(0, len(vocab), size=5)

# Words of wisdom as a list of tokens
random_words = [idx2word[i] for i in randidx]
print("Random words as list:", random_words)

# Does it sound more wise as text??
random_sentence = ' '.join([idx2word[i] for i in randidx])
print(f'Random "wisdom": "{random_sentence}"')
print()

# ============================================
# A peek at tokenization
# ============================================

# Translate the text into numbers
text_as_int = [word2idx[word] for word in allwords]
print("First 10 tokens as integers:")
print(text_as_int[:10])
print()

# And numbers back into text
print("First 10 tokens decoded:")
for tokeni in text_as_int[:10]:
    print(f'  Token {tokeni:2}: "{idx2word[tokeni]}"')
print()


# ============================================
# Exercise 2: Wrap the encoder/decoder into functions
# ============================================

### The encoder function
def encoder(text, word2idx):
    """Convert text string to list of token IDs"""
    # Parse the text into words (split and lowercase)
    words = re.split('\s', text.lower())

    # Return the vector of indices
    # Handle unknown words with a default index (-1)
    return [word2idx.get(word, -1) for word in words if word]  # Skip empty strings


### Now for the decoder
def decoder(indices, idx2word):
    """Convert list of token IDs back to text"""
    # Find the words for these indices, and join into one string
    # Handle unknown indices with <UNK> token
    return ' '.join([idx2word.get(idx, '<UNK>') for idx in indices])


# Reminder of the available words
print("=" * 50)
print("Exercise 2: Encoder/Decoder Functions")
print("=" * 50)
print(f"Available vocabulary ({len(vocab)} words):")
print(vocab)
print()

# Create a new sentence using the vocab
newtext = 'we already are the result of what everyone else already thought'

# Encode and decode the new text
newtext_tokenIDs = encoder(newtext, word2idx)
decoded_text = decoder(newtext_tokenIDs, idx2word)

print('Original text:')
print(f'\t{newtext}')

print(f'\nToken IDs:')
print(f'\t{newtext_tokenIDs}')

print(f'\nDecoded text:')
print(f'\t{decoded_text}')
print()

# ============================================
# Exercise 3: Visualize the tokens
# ============================================

print("=" * 50)
print("Exercise 3: Token Visualization")
print("=" * 50)

# Get all the text and all the tokens
alltext = ' '.join(vocab)
tokens = encoder(alltext, word2idx)

# Create a figure
fig, ax = plt.subplots(1, figsize=(14, 6))

# Plot the tokens
ax.plot(range(len(tokens)), tokens, 'ks', markersize=12, markerfacecolor=[.7, .7, .9])
ax.set(xlabel='Word Position', ylabel='Token ID',
       title='Vocabulary Token IDs')
ax.grid(linestyle='--', axis='y', alpha=0.3)

# Invisible axis for right-hand-side labels
ax2 = ax.twinx()
ax2.plot(tokens, alpha=0)
ax2.set(yticks=range(len(vocab)), yticklabels=vocab)
ax2.set_ylabel('Word', rotation=270, labelpad=20)

# Add some word labels on the plot for clarity
for i, (token, word) in enumerate(zip(tokens[:10], vocab[:10])):
    ax.annotate(word, (i, token), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=8, rotation=45)

plt.tight_layout()
plt.show()

# Additional visualization: Token distribution in original texts
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 8))

# Tokenize each original sentence
for i, sentence in enumerate(text):
    sentence_tokens = encoder(sentence, word2idx)
    ax3.plot(sentence_tokens, 'o-', label=f'Text {i + 1}', alpha=0.7, linewidth=2)

ax3.set(xlabel='Word Position in Sentence', ylabel='Token ID',
        title='Token IDs for Original Sentences')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Word frequency histogram
word_counts = {}
for word in allwords:
    word_counts[word] = word_counts.get(word, 0) + 1

sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
words_plot = [w[0] for w in sorted_words]
counts_plot = [w[1] for w in sorted_words]

bars = ax4.bar(range(len(words_plot)), counts_plot, color='steelblue', alpha=0.7)
ax4.set(xlabel='Word', ylabel='Frequency',
        title='Top 10 Most Frequent Words')
ax4.set_xticks(range(len(words_plot)))
ax4.set_xticklabels(words_plot, rotation=45, ha='right')

# Add count values on bars
for bar, count in zip(bars, counts_plot):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2., height,
             f'{count}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Summary statistics
print("\n" + "=" * 50)
print("Summary Statistics")
print("=" * 50)
print(f"Total words processed: {len(allwords)}")
print(f"Unique words (vocab size): {len(vocab)}")
print(f"Most common word: '{sorted_words[0][0]}' (appears {sorted_words[0][1]} times)")
print(f"Vocabulary coverage: {len(vocab) / len(allwords) * 100:.1f}% unique words")