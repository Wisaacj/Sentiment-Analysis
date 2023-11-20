import os
import numpy as np
from typing_extensions import Self
from functools import cached_property
from sklearn.model_selection import StratifiedShuffleSplit


class TextualDataPoint:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.contents: str | list[str] = self._extract_contents()
        """The contents of the datapoint; initially a string, and then a list of tokens
        after the datapoint has been tokenized."""

    @cached_property
    def basename(self) -> str:
        return os.path.basename(self.file_path)

    @cached_property
    def filename(self) -> str:
        return os.path.splitext(self.basename)[0]
    
    def _extract_contents(self) -> str:
        with open(self.file_path, "r") as file:
            return file.read()
        
    def as_dict(self) -> dict:
        return {
            'contents': self.contents,
        }


class Review(TextualDataPoint):

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.id = self._parse_id()
        self.rating = self._parse_rating()
        self.polarity = self._determine_polarity()

    def _parse_id(self) -> int:
        return self.filename.split("_")[0]

    def _parse_rating(self) -> int:
        return int(self.filename.split("_")[1])
    
    def _determine_polarity(self) -> int:
        if self.rating is None:
            self.rating = self._parse_rating()

        if self.rating <= 4:
            return 0
        elif self.rating >= 7:
            return 1
        else:
            raise ValueError(f"unexpected rating: {self.rating}")

    def as_dict(self) -> dict:
        return {
            'id': self.id,
            'rating': self.rating,
            'polarity': self.polarity,
        } | super().as_dict()
    

class IterableSet:

    datapoint_class = TextualDataPoint

    def __init__(self, datapoints: list[datapoint_class]):
        self.datapoints = datapoints

        # To keep track of the current iteration position.
        self.index = 0

    def first(self) -> datapoint_class:
        return self.datapoints[0]

    def last(self) -> datapoint_class:
        return self.datapoints[-1]
    
    def __len__(self) -> int:
        return len(self.datapoints)
    
    def __iter__(self) -> Self:
        # Reset the index whenever starting a new iteration.
        self.index = 0
        return self
        
    def __next__(self) -> datapoint_class:
        # Make sure there are more datapoints to yield.
        if self.index < len(self.datapoints):
            result = self.datapoints[self.index]
            self.index += 1
            return result
        else:
            # No more datapoints -> raise StopIteration exception.
            raise StopIteration

    def as_lower_representation(self) -> list[dict]:
        return [
            datapoint.as_dict()
            for datapoint in self.datapoints
        ]


class DataSet(IterableSet):

    def __init__(self, dirs: list[str]):
        super().__init__(None)
        self.dirs = dirs
        
    def load(self) -> Self:
        self.datapoints = [
            self.datapoint_class(directory + file)
            for directory in self.dirs
            for file in os.listdir(directory)
        ]

        return self
    
    def as_lower_representation(self) -> list[dict]:
        # Ensure the dataset has been loaded.
        if self.datapoints is None:
            self.load()

        return super().as_lower_representation()

    def __iter__(self) -> Self:
        # Ensure the dataset has been loaded.
        if self.datapoints is None:
            self.load()

        return super().__iter__()
    

class ReviewDataSet(DataSet):

    datapoint_class = Review


class FeatureSet(IterableSet):

    def __init__(self, dataset: DataSet):
        super().__init__(dataset.datapoints)

    def compare_with(self, other_set: Self):
        set1_dp1 = self.first().contents
        set2_dp1 = other_set.first().contents
        max_length_set1 = len(max(set1_dp1, key=len))

        print("Comparing the first datapoint in feature sets A and B respectively:")
        for token1, token2 in zip(set1_dp1, set2_dp1):
            empty_space = " " * (max_length_set1 - len(token1))
            print(f"Set A: {token1} {empty_space}| Set B: {token2}")

    def as_inputs_and_targets(self, target_variable_name: str):
        inputs = [datapoint.contents for datapoint in self.datapoints]
        targets = [getattr(datapoint, target_variable_name)
                   for datapoint in self.datapoints]

        return np.array(inputs), np.array(targets)

    def split_into_train_dev_test_sets(self, target_variable_name: str, dev_test_size: float, random_state: int = 42):
        inputs, targets = self.as_inputs_and_targets(target_variable_name)

        # Split the data into train and dev+test sets in a ratio of (1-dev_test_size):(dev_test_size).
        initial_splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=dev_test_size, random_state=random_state)
        train_indexes, test_indexes = next(
            initial_splitter.split(inputs, targets))

        X_train, y_train = inputs[train_indexes], targets[train_indexes]
        X_test_dev, y_test_dev = inputs[test_indexes], targets[test_indexes]

        # Split the dev+test set into dev and test sets in a 50:50 ratio.
        final_splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.5, random_state=random_state)
        dev_indexes, test_indexes = next(
            final_splitter.split(X_test_dev, y_test_dev))

        X_dev, y_dev = X_test_dev[dev_indexes], y_test_dev[dev_indexes]
        X_test, y_test = X_test_dev[test_indexes], y_test_dev[test_indexes]

        return X_train, y_train, X_dev, y_dev, X_test, y_test