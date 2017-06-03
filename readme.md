# Neural Paraphrase Identification of Questions with Noisy Pretraining
This is an attempt to reproduce the paper using tensorflow.

## Unclear parts
How are short words processed? We just suppose <5 n-gram have a null character

Is there still a null word at the start? We suppose not.

Are there OOV n-gram characters? (it's possible that some words have no n-grams at all), We suppose this is no since paper does not mention anything about OOV, and that these sequences have a zero vector.

We do not include null or OOV words since these are not mentioned in the paper. For documents with no words (once OOV words removed), we sample a random word uniformly to avoid model issues.

What is the context size? We suppose it is the intra-sentence attention context size.

We suppose there is no projection and word embeddings are trained.

XXX do DECATTword first, and do same model as first paper and then try alternative model
