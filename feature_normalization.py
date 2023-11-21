import copy
import math
import numpy as np
from collections import Counter, defaultdict

from dataset_types import FeatureSet


class FeatureSetNormalizer:

    def __init__(self, feature_set: FeatureSet):
        # We don't want to modify the original feature set.
        self.feature_set = copy.deepcopy(feature_set)

        self.normalized = False
        self.shared_vocabulary = self._collect_shared_vocabulary()

    def perform_tf(self) -> FeatureSet:
        self._calculate_term_frequencies()

        for datapoint in self.feature_set:
            datapoint.contents = [
                datapoint.term_frequencies.get(token, 0)
                for token in self.shared_vocabulary
            ]

        return self.feature_set

    def perform_tf_idf(self) -> FeatureSet:
        self._calculate_term_frequencies()
        self.idfs = self._calculate_idfs()

        for datapoint in self.feature_set:
            datapoint.contents = [
                (datapoint.term_frequencies.get(token, 0) * self.idfs.get(token))
                for token in self.shared_vocabulary
            ]

        return self.feature_set

    def peform_ppmi(self) -> FeatureSet:
        raise NotImplementedError

    def _collect_shared_vocabulary(self) -> set:
        return set(sorted({
            token
            for datapoint in self.feature_set
            for token in datapoint.contents
        }))
    
    def _calculate_idfs(self) -> dict:
        # We calculate the document frequencies by creating a unique set of tokens for
        # each datapoint (i.e., for each document in the set, counting each token once
        # per document regardless of its frequency within the document itself). The Counter
        # then aggregates these sets across all datapoints, counting the number of documents
        # in which each distinct token appears. This gives us the document frequency for
        # each term in the `shared_vocabulary`.
        document_frequencies = Counter(
            token
            for datapoint in self.feature_set
            for token in set(datapoint.contents)
        )
        num_datapoints = len(self.feature_set)

        return {
            token: math.log(num_datapoints / (doc_frequency + 1))
            for token, doc_frequency in document_frequencies.items()
        }
    
    def _calculate_term_frequencies(self) -> None:
        for datapoint in self.feature_set:
            num_tokens = len(datapoint.contents)
            term_occurences = Counter(datapoint.contents)
            
            # Normalise the term occurences by dividing them by the length of the datapoint.
            datapoint.term_frequencies = {
                token: count / num_tokens
                for token, count in term_occurences.items()
            }

    def perform_fast_tf_idf(self) -> FeatureSet:
        vocab_to_index = {word: idx for idx, word in enumerate(self.shared_vocabulary)}
        num_documents = len(self.feature_set)
        num_vocab = len(self.shared_vocabulary)
        
        # TF matrix where each row corresponds to a document, and each column corresponds to a term.
        tf_matrix = np.zeros(shape=(num_documents, num_vocab), dtype=float)
        
        # Document frequency (DF) counter for counting in how many documents a term appears.
        df_counter = defaultdict(int)
        
        # Populate TF matrix and DF counter
        for doc_idx, datapoint in enumerate(self.feature_set):
            # Count term occurrences in the document
            term_occurrences = Counter(datapoint.contents)
            for term, count in term_occurrences.items():
                if term in vocab_to_index:
                    index = vocab_to_index[term]
                    tf_matrix[doc_idx, index] = count  # Raw count for TF (to be normalized later)
                    df_counter[term] += 1
        
        # Normalize TF matrix row-wise (divide by the number of terms in each document)
        doc_lengths = np.array([len(dp.contents) for dp in self.feature_set])
        tf_matrix = tf_matrix / doc_lengths[:, None]  # Broadcasting division
        
        # Convert the DF counter into an array of IDF values
        # idf_array = np.log((1 + num_documents) / (1 + np.array([df_counter[term] for term in self.shared_vocabulary]))) + 1
        idf_array = np.log((num_documents) / (1 + np.array([df_counter[term] for term in self.shared_vocabulary])))
        
        # TF-IDF calculation by multiplying the TF matrix by the IDF values
        # The transpose on idf_array is necessary for broadcasting to correct dimension
        tfidf_matrix = tf_matrix * idf_array
        
        # Update datapoint contents with tf-idf values
        for doc_idx, datapoint in enumerate(self.feature_set):
            datapoint.contents = tfidf_matrix[doc_idx, :] # .tolist()

        return self.feature_set