# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause
# CoAuthor: Giovanni Cavallin

from pprint import pprint
from time import time
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

categories = [
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
]

data_train = fetch_20newsgroups(subset='train', categories=categories)
data_test = fetch_20newsgroups(subset='test', categories=categories)

# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    # ('clf', SGDClassifier())
    ('clf', MultinomialNB()),
    # ('clf', BernoulliNB())
])

parameters = {
    'vect__max_df': (0.75, 1.0),  # deletes the words that are present in 50%, 75% and 100% of the present documents
    'vect__max_features': (None, 5000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3), ),  # unigrams or bigrams
    # 'tfidf__use_idf': (True, False),  # is true by default and is good
    # 'tfidf__norm': ('l2', 'l1'),  # is l2 by default and is good for me
    # 'clf__max_iter': (5,),
    'clf__alpha': (0.05, 0.1, 0.15),  # good for SGD but smaller - default: 1e-5
    # 'clf__penalty': ('l2', 'elasticnet'),  # to use with SGD
    # 'clf__max_iter': (200, ),  # to use with SGD
    'vect__stop_words': (stop_words, ),
}

if __name__ == "__main__":
    grid_search = GridSearchCV(pipeline, parameters, cv=5,
                               n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(data_train.data, data_train.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    y_pred = grid_search.predict(data_test.data)
    y_test = data_test.target

    print('Report: ', metrics.classification_report(y_test, y_pred))
    print('Confusion matrix: ', metrics.confusion_matrix(y_test, y_pred))
    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average='macro')
    print('Accuracy: ', acc)
    print('F1: ', f1)
