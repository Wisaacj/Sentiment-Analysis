import copy
import math
from collections import Counter
from functools import cached_property

from dataset_types import FeatureSet


class FeatureSetNormalizer:

    def __init__(self, feature_set: FeatureSet):
        # We don't want to modify the original feature set.
        self.feature_set = copy.deepcopy(feature_set)

        self.normalized = False
        self.shared_vocabulary = self._collect_shared_vocabulary()

    @cached_property
    def num_datapoints(self) -> int:
        return len(self.feature_set)

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

        return {
            token: math.log(self.num_datapoints / (doc_frequency + 1))
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
