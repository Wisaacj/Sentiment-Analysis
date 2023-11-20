import copy
import string
from typing_extensions import Self

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams, everygrams

from dataset_types import DataSet, FeatureSet


class Preprocessor:

    def __init__(self, dataset: DataSet):
        # We don't want to modify the original dataset.
        self.dataset = copy.deepcopy(dataset)
        # Tokenization is the first preprocessing step of most NLP applications.
        self.tokenize()

    def tokenize(self) -> Self:
        for datapoint in self.dataset:
            if isinstance(datapoint.contents, list):
                # This datapoint has already been tokenized.
                continue

            datapoint.contents = nltk.word_tokenize(datapoint.contents)
        
        return self
    

class FeatureSetGenerator(Preprocessor):

    def create_n_grams(self, n: int) -> FeatureSet:
        for datapoint in self.dataset:
            datapoint.contents = list(ngrams(datapoint.contents, n))

        return FeatureSet(self.dataset)
    
    def create_everygrams(self, max_n: int) -> FeatureSet:
        for datapoint in self.dataset:
            datapoint.contents = list(everygrams(datapoint.contents, max_len=max_n))

        return FeatureSet(self.dataset)
    
    def to_lowercase(self) -> Self:
        for datapoint in self.dataset:
            datapoint.contents = [token.lower() for token in datapoint.contents]

        return self
    
    def remove_stopwords(self) -> Self:
        distinct_stopwords = set(stopwords.words('english'))

        for datapoint in self.dataset:
            datapoint.contents = [token for token in datapoint.contents if token not in distinct_stopwords]

        return self
    
    def remove_punctuation(self) -> Self: 
        for datapoint in self.dataset:
            datapoint.contents = [token for token in datapoint.contents if token not in string.punctuation]

        return self

    def lemmatize(self) -> Self:
        lmtzr = WordNetLemmatizer()

        for datapoint in self.dataset:
            datapoint.contents = [lmtzr.lemmatize(token) for token in datapoint.contents]

        return self

    def stem(self) -> Self:
        # Making the assumption that all datapoints are in English.
        stmr = SnowballStemmer("english")

        for datapoint in self.dataset:
            datapoint.contents = [stmr.stem(token) for token in datapoint.contents]

        return self