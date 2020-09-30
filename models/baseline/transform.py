from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, wordpunct_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
import unicodedata

class Tokenize(BaseEstimator, TransformerMixin):
    def tokenize(self, sentence):
        words = wordpunct_tokenize(sentence)
        tokens = pos_tag(words)
        return tokens

    def transform(self, x):
        return [self.tokenize(review) for review in x]

    def fit(self, x, y=None):
        return self

class Normalize(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords  = set(stopwords.words('english'))

    def _is_punct(self, token):
        return all([unicodedata.category(char).startswith('P') for char in token[0]])

    def _is_stopword(self, token):
        return token[0].lower() in self.stopwords

    def _lemanizer(self, token):
        word, tag = token
        tag = {"N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV, "J": wordnet.ADJ}.get(tag[0], wordnet.NOUN)
        return self.lemmatizer.lemmatize(word, tag)

    def transform(self, x):
        for tkzd_review in x:
            ntokens = []
            for token in tkzd_review:
                if self._is_punct(token) or self._is_stopword(token):
                    continue
                else:
                    ntokens.append(self._lemanizer(token).lower())
            yield ntokens

    def fit(self, x, y=None):
        return self
