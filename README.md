# Duality
##### Morino
the kernel part for a virtual character

1. First setup the package dependancy a.t. _./setup.py_
2. See https://spacy.io/docs/usage/ for a further configuration of the spacy's corpus
3. See http://www.nltk.org/data.html about how to download corpus _stopwords_ [only this part of nltk is used]
4. Modify the DIR in the _./context.py_ as /path/to/tweet140
5. Run a rough preprocessor
>  python3 -m duality.preproc
6. Run the vocabulary builder and the verb extractor samples
>  python3 -m duality.models
