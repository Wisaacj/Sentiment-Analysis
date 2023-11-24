import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod


class IClassifier(ABC):

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError


class NaiveBayesClassifier(IClassifier):

    def __init__(self, k: int = 2, alpha: float = 1):
        """
        
        :param k: The number of classes in the classification problem.
        :param alpha: A laplace smoothing parameter.
        """
        self.k = k
        self.alpha = alpha

    def fit(self, data: npt.NDArray, labels: npt.NDArray):
        num_samples, num_features = data.shape

        # Allocate memory for the class prior probabilities and class conditional feature likelihoods.
        self.class_prior_probas = np.zeros(shape=(self.k,))
        self.class_conditional_feature_likelihoods = np.zeros(shape=(self.k, num_features))

        for cls in range(self.k):
            samples_of_class = data[labels == cls]
            class_sample_count = samples_of_class.shape[0]

            # Probability class cls occurs. P(Y)
            self.class_prior_probas[cls] = class_sample_count / num_samples

            # Term frequencies for class `cls`.
            cls_global_feature_totals = samples_of_class.sum(axis=0) # shape = (num_features, )

            # Compute the conditional probability for each feature given the class
            # (including Laplace smoothing). P(X | Y)
            self.class_conditional_feature_likelihoods[cls, :] = (
                    (cls_global_feature_totals + self.alpha) /
                    (np.sum(cls_global_feature_totals) + self.alpha * num_features)
                )

    def predict(self, data: npt.NDArray) -> npt.NDArray:
        # Calculate the log of the priors.
        log_class_priors = np.log(self.class_prior_probas)

        # Calculate the log probabilities of each class for each sample.
        logits = log_class_priors + (data @ np.log(self.class_conditional_feature_likelihoods).T)

        # For each sample, get the index of the maximum logit. These are the predicted classes.
        return np.argmax(logits, axis=1)

    def legacy_predict(self, data: npt.NDArray):
        num_samples, _ = data.shape

        # Allocate memory for our predictions.
        predictions = np.zeros(shape=(num_samples,))

        # Calculate the log of class prior probabilities.
        log_class_priors = np.log(self.class_prior_probas)

        for idx, sample in enumerate(data):
            # Log probabilities of each class for the current sample.
            logits = np.zeros(shape=(self.k,))

            for cls in range(self.k):
                # The log probability of each feature given the class.
                feature_log_probabilities = sample * np.log(self.class_conditional_feature_likelihoods[cls, :])

                # Sum the feature log probabilties to get the total log probability of the sample
                # given the class.
                log_prob_features_given_class = np.sum(feature_log_probabilities)

                # Add the prior probability of the class itself to get the final log probability
                # for the class.
                logits[cls] = log_class_priors[cls] + log_prob_features_given_class

            predictions[idx] = np.argmax(logits)

        return predictions