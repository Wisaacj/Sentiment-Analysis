import numpy as np
import pandas as pd
import numpy.typing as npt
from string import ascii_uppercase

from classifiers import IClassifier


class ClassifierPerformanceSummary:

    def __init__(self, accuracy: float, precision: float, recall: float, f1: float):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1

    def __repr__(self) -> str:
        return f"""
        Accuracy:  {self.accuracy*100:.4f}%
        Precision: {self.precision*100:.4f}%
        Recall:    {self.recall*100:.4f}%
        F1:        {self.f1*100:.4f}%
        """
    
    def as_dict(self) -> dict:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1
        }

    def as_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.as_dict(), index=[0])


class BinaryClassifierEvaluator:

    def __init__(self, ground_truth: npt.NDArray, predictions: npt.NDArray):
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.confusion_matrix = self.produce_confusion_matrix()
        """
        [[ TP, FP ],
         [ FN, TN ]]
        """

    def produce_confusion_matrix(self) -> npt.NDArray:
        confusion_matrix = np.zeros(shape=(2, 2))

        # True positives: where both prediction and ground truth are 1.
        confusion_matrix[0, 0] = np.sum((self.predictions == 1) & (self.ground_truth == 1))

        # False positives: where prediction is 1, but ground truth is 0.
        confusion_matrix[0, 1] = np.sum((self.predictions == 1) & (self.ground_truth == 0))
        
        # False negatives: where prediction is 0, but ground truth is 1.
        confusion_matrix[1, 0] = np.sum((self.predictions == 0) & (self.ground_truth == 1))
        
        # True negatives: where both prediction and ground truth are 0.
        confusion_matrix[1, 1] = np.sum((self.predictions == 0) & (self.ground_truth == 0))

        return confusion_matrix.astype(int)

    def calculate_accuracy(self) -> float:
        """
        Calculates (TP + TN) / Num Predictions.
        """
        return (self.confusion_matrix[0, 0] + self.confusion_matrix[1, 1]) / self.confusion_matrix.sum()

    def calculate_recall(self) -> float:
        """
        Calculates TP / (TP + FN)
        """
        return self.confusion_matrix[0, 0] / self.confusion_matrix[:, 0].sum()

    def calculate_precision(self) -> float:
        """
        Calculates TP / (TP + FP).
        """
        return self.confusion_matrix[0, 0] / self.confusion_matrix[0, :].sum()
    
    def calculate_f1(self) -> float:
        """
        Calculates the harmonic mean of precision and recall.
        """
        precision = self.calculate_precision()
        recall = self.calculate_recall()

        return 2 * (precision * recall) / (precision + recall)
    
    def get_summary(self) -> ClassifierPerformanceSummary:
        return ClassifierPerformanceSummary(
            self.calculate_accuracy(),
            self.calculate_precision(),
            self.calculate_recall(),
            self.calculate_f1()
        )
    

class BaseComparator:
    """Base Comparator Class."""

    def __init__(self, X_trains: list[npt.NDArray], y_trains: list[npt.NDArray]):
        """
        Initialises a comparator. The length of X_trains must equal the length of y_trains.

        :param X_trains: a list of training datasets.
        :param y_trains: a list of training label sets.
        """
        self.X_trains = X_trains
        self.y_trains = y_trains


class FeatureSetComparator(BaseComparator):
    """Class for comparing feature sets."""

    def train_and_evaluate(
            self, 
            cls: IClassifier, 
            X_train: npt.NDArray, 
            y_train: npt.NDArray, 
            X_test: npt.NDArray, 
            y_test: npt.NDArray, 
            hyperparams: dict
            ) -> pd.DataFrame:
        
        classifier: IClassifier = cls(**hyperparams)
        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)
        return BinaryClassifierEvaluator(y_test, predictions).get_summary().as_dict()
    
    def compare(
            self, 
            classifier_cls: IClassifier, 
            X_tests: list[npt.NDArray], 
            y_tests: list[npt.NDArray], 
            hyperparams: dict
            ) -> pd.DataFrame:
        
        performance_data = pd.DataFrame()
        for i in range(len(self.X_trains)):
            row_index = f"{classifier_cls.__name__} - Set {ascii_uppercase[i]}"
            performance_data[row_index] = self.train_and_evaluate(
                classifier_cls,
                self.X_trains[i],
                self.y_trains[i],
                X_tests[i],
                y_tests[i],
                hyperparams
            )

        return performance_data.T


class ClassifierComparator(BaseComparator):
    """Class for comparing classifiers across multiple feature sets."""

    def compare(
            self, 
            classifier_classes: list[IClassifier], 
            X_tests: list[npt.NDArray], 
            y_tests: list[npt.NDArray]
            ) -> pd.DataFrame:
        
        classifier_data = pd.DataFrame()
        for cls in classifier_classes:
            performance = FeatureSetComparator(self.X_trains, self.y_trains)\
                .compare(cls, X_tests, y_tests, {})
            
            classifier_data = pd.concat([classifier_data, performance], axis=0)

        return classifier_data