# Plan of Action

## [ ] Lab work
Complete module labs up to week 4 (or 5)
- [ ] AuthorProfiling
- [ ] AnalogySolver (opt)
- [ ] WordNetPractice (opt)

## [x] Feature Generation
Generate at least three feature sets using paths through the following set of choices:
- Tokenization (must)
- Lemmatization OR Stemming
- Lowercase (or not)
- Remove stopwords (or not)
- Remove punctuation (or not)
- N-gram generation (pick 1, 2, 3, ..., n)
- Normalisation methods
    - Frequency Normalisation
    - Tf-Idf
    - PPMI

## [ ] Naive Bayes
- [x] Implement Naive Bayes from scratch
- [ ] Evaluate it on each of the three feature sets you've extracted (using the development splits)
- [ ] Pick the best of the three feature sets you've experimented with and present the test split results
- [ ] Evaluate the `scikit-learn` implementation of Naive Bayes on the same three feature sets and compare the results to your implementation

## [ ] SGD-based classification and SVMs
For this task, you can use `scikit-learn`'s implementation of the following models:

Logistic Regression
- [x] Train a Logistic Regression classifier
- [ ] Evaluate each model on each of the three feature sets (using the respective development splits)
- [ ] Pick the best of the three feature sets you've experimented with and present the test split results

SVM
- [x] Train a SVM classifier
- [ ] Evaluate each model on each of the three feature sets (using the respective development splits)
- [ ] Pick the best of the three feature sets you've experimented with and present the test split results

Subsequently, you should perform hyperparameter optimisation on your best performing model/feature set.
- [ ] Try at least 5 different combinations of hyperparameters using the development split
    - Consider using more comprehensive hyperparameter optimisation techniques like gridsearch or k-fold validation.
- [ ] Evaluate the best hyperparameters using the test split

_Note that you should use 1-hot embeddings based on your features for input._

## [ ] BERT
- [ ] Follow the given tutorials and train BERT to perform sentiment analysis over the same train/dev/test splits of the feature sets.
- [ ] Experiment with both the cased and uncased version of BERT

## [ ] ChatGPT
**TODO**: Come up with a plan.