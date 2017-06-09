# Neural Paraphrase Identification of Questions with Noisy Pretraining
This is an attempt to reproduce the paper using tensorflow.

## Unclear parts
How are short words processed? We just suppose <5 n-gram have a null character

Is there still a null word at the start? We suppose not.

Are there OOV n-gram characters? (it's possible that some words have no n-grams at all), We suppose this is no since paper does not mention anything about OOV, and that these sequences have a zero vector.

We suppose the context window for "abc" is "00a", "0ab", "abc", "bc0", "c00".

We do not include null words since these are not mentioned in the paper. We include OOV words of GloVe since even rare words appear often in the validation and test sets due to question overlap. Even if not the case probably will do better due to use of trigrams instead of just words since OOV words should be part of the trigram. OOV words in validation and test are thus untrained.

It is unclear what the intra-sentence bias is.

We suppose there is no projection and word embeddings are trained.
