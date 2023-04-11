[Youtube](https://www.youtube.com/watch?v=iYCr1fQ4eVk)

### TF-IDF

- a document that contains 100 words and out of these 100 words 'research' appears 5 times. Then _term frequency_ will be 5/100 = 0.05
- there are 40000 documents and out of these only 400 documents contains 'research'. Then the _inverse document frequency_ IDF(research) = (40000/400) = 100. TF-IDF(research) = 0.05 \* 100 = 5
- Terminoology:
  - t - _term_ is a word,
  - d - _document_(set of words),
  - N - count of _corpus_
  - corpus - the total document set

### Data preprocessing

- missing values
- tokenization
- normalization
- stemming
- lemmatization
- removing stop words
- noise removal
