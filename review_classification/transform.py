from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
import unicodedata

class Normalize(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords  = set(stopwords.words("english"))

    def is_punct(self, token):
        return all([unicodedata.category(char).startswith("P") for char in token[0]])

    def is_stopword(self, token):
        return token[0].lower() in self.stopwords

    def lemanizer(self, token):
        word, tag = token
        tag = {"N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV, "J": wordnet.ADJ}.get(tag[0], wordnet.NOUN)
        return self.lemmatizer.lemmatize(word, tag)

    def transform(self, document):
        # document := [[word11, word12, ...], [word21, word22, ...], ...]
        # tokenX := (wordX, POSX)
        for review in document:
            ntokens = []
            for token in review:
                if self.is_punct(token) or self.is_stopword(token):
                    continue
                else:
                    ntokens.append(self.lemanizer(token).lower())
            yield ntokens

    def fit(self, X, y=None):
        return self
