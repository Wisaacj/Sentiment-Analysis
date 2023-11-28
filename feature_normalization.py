import copy
import math
from typing import Tuple
import numpy as np
import numpy.typing as npt
from collections import Counter, defaultdict

from dataset_types import FeatureSet


class FeatureSetNormalizer:

    def __init__(self, feature_set: FeatureSet):
        # We don't want to modify the original feature set.
        self.feature_set = copy.deepcopy(feature_set)

        self.normalized = False
        self.shared_vocabulary = self._collect_shared_vocabulary()

        self.num_samples = len(self.feature_set)
        self.num_features = len(self.shared_vocabulary)

    def perform_tf_norm(self, drop_percentile: float = 0) -> FeatureSet:
        _, tf_matrix = self._calculate_tf_idf_scores(drop_percentile)

        for doc_idx, datapoint in enumerate(self.feature_set):
            datapoint.contents = tf_matrix[doc_idx, :]

        return self.feature_set

    def perform_tf_idf_norm(self, drop_percentile: float = 0) -> FeatureSet:
        tfidf_matrix, _ = self._calculate_tf_idf_scores(drop_percentile)

        # Update datapoint contents with tf-idf values
        for doc_idx, datapoint in enumerate(self.feature_set):
            datapoint.contents = tfidf_matrix[doc_idx, :]

        return self.feature_set

    def peform_ppmi(self) -> FeatureSet:
        raise NotImplementedError

    def _collect_shared_vocabulary(self) -> set:
        return {
            token
            for datapoint in self.feature_set
            for token in datapoint.contents
        }
    
    def _remove_rare_features(self, tfidf_matrix: npt.NDArray, tf_matrix: npt.NDArray, drop_percentile: float) -> Tuple[npt.NDArray, npt.NDArray]:
        total_tfidf_per_feature = np.sum(tfidf_matrix, axis=0)
        total_tfidf = np.sum(total_tfidf_per_feature)

        # Get the indices of total_tfidf_per_feature if it were sorted.
        sorted_indices = np.argsort(total_tfidf_per_feature)
        # Calculate the cumulative sum along the sorted features.
        sorted_cumulative_tfidf = np.cumsum(total_tfidf_per_feature[sorted_indices])

        # Determine the cut-off index where the cumulative sum reaches the threshold
        # percentage.
        threshold_index = np.searchsorted(sorted_cumulative_tfidf, drop_percentile * total_tfidf)

        # Use the threshold_index to determine the indices of features to keep.
        features_to_keep_indices = sorted_indices[threshold_index:]

        # Keep only the columns for features we want to retain.
        tfidf_matrix = tfidf_matrix[:, features_to_keep_indices]
        tf_matrix = tf_matrix[:, features_to_keep_indices]

        return tfidf_matrix, tf_matrix
    
    def _calculate_tf_idf_scores(self, drop_percentile: float) -> Tuple[npt.NDArray, npt.NDArray]:
        vocab_to_index = {word: idx for idx, word in enumerate(self.shared_vocabulary)}

        # Term frequency matrix. Each row corresponds to a document and each column to a term.
        tf_matrix = np.zeros(shape=(self.num_samples, self.num_features), dtype=float)
        # This will store a count of how many documents each term appears in, defaulting to 0.
        df_counter = defaultdict(int)

        # Populate tf_matrix and df_counter
        for doc_idx, datapoint in enumerate(self.feature_set):
            # Count term occurences in this document.
            term_occurences = Counter(datapoint.contents)
            for term, count in term_occurences.items():
                if term in vocab_to_index:
                    index = vocab_to_index[term]
                    # Raw count for TF (to be normalised later)
                    tf_matrix[doc_idx, index] = count
                    # Increment the df counter.
                    df_counter[term] += 1

        # Normalise the term frequency matrix row-wise (divide by the number of terms in each document).
        doc_lengths = np.array([len(datapoint.contents) for datapoint in self.feature_set])
        tf_matrix = tf_matrix / doc_lengths[:, None]

        # Transform document frequencies into inverse-document frequencies.
        idf_array = np.log(
            (self.num_samples) / (1 + np.array([df_counter[term] for term in self.shared_vocabulary]))
        )

        # Calculate the TF-IDF matrix by multily the TF matrix by the IDF values.
        tfidf_matrix = tf_matrix * idf_array

        if drop_percentile > 0:
            # Remove features that appear most infrequently.
            tfidf_matrix, tf_matrix = self._remove_rare_features(tfidf_matrix, tf_matrix, drop_percentile)

        return tfidf_matrix, tf_matrix