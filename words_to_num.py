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


text=["All that we are is the result of what we thought,"
      "To be or not to be that is the question",
      "Be yourself everyone else is already taken"
      ]
print(text)

#seperate into word by spletting by spaces
import re
re.split('\s',text[0])
#can recobine into text
''.join(re.split('\s',text[0]))

# also make lower-case
allwords = re.split('\s',' '.join(text).lower())
print(allwords)


#Create a vocabulary (lexicon)

# find the unique words
vocab = sorted(set(allwords))
print(vocab)


#Create an encoder and decoder

# the encoder is a python dictionary type
word2idx = {}
for i, word in enumerate(vocab):
    word2idx[word] = i
word2idx

# and a decoder
idx2word = {}
for i, word in enumerate(vocab):
    idx2word[i] = word
idx2word

print(f'The word "to" has index {word2idx["to"]}')
print(f'The index "7" maps to the word "{idx2word[7]}"')


#Make fake quotes, just for fun :P


# select random words from the dictionary
import numpy as np

randidx = np.random.randint(0, len(vocab), size=5)

# words of wisdom as a list of tokens
[idx2word[i] for i in randidx]

# does it sound more wise as text??
' '.join([idx2word[i] for i in randidx])


#A peak at tokenization

# translate the text into numbers
text_as_int = [word2idx[word] for word in allwords]
text_as_int

# and numbers back into text
for tokeni in text_as_int:
    print(f'Token {tokeni:2}: {idx2word[tokeni]}')
