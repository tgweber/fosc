################################################################################
# Copyright: Tobias Weber 2022
#
# Apache 2.0 License
#
# This file contains all persistance-related tests
#
################################################################################

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_classif, SelectKBest

from fosc import \
    dump_vectorizer_to_dir,\
    load_vectorizer_from_dir,\
    dump_selector_to_dir,\
    load_selector_from_dir


def test_store_and_load_vectorizer(tmpdir):
    corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?',
    ]
    vectorizer = TfidfVectorizer(stop_words=["This"], norm="l1")
    vectorizer.fit(corpus)

    dump_dir = tmpdir.mkdir("test-persistence").realpath()
    dump_vectorizer_to_dir(vectorizer, dump_dir)
    vectorizer_loaded = load_vectorizer_from_dir(dump_dir)
    assert np.array_equal(vectorizer.idf_, vectorizer_loaded.idf_)
    for key, value in vectorizer_loaded.vocabulary_.items():
        assert vectorizer.vocabulary_[key] == value
    for key, value in vectorizer.vocabulary_.items():
        assert vectorizer_loaded.vocabulary_[key] == value
    for key, value in vectorizer_loaded.get_params().items():
        assert vectorizer.get_params()[key] == value
    for key, value in vectorizer.get_params().items():
        assert vectorizer_loaded.get_params()[key] == value

def test_store_and_load_selector(tmpdir):
    corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?',
    ]
    vectorizer = TfidfVectorizer(stop_words=["This"], norm="l1")
    x_all = vectorizer.fit_transform(corpus)
    selector = SelectKBest(f_classif, k=4).fit(x_all, [1,2,1,2])
    x = selector.transform(x_all).astype(np.float64)
    dump_dir = tmpdir.mkdir("test-persistence").realpath()
    dump_selector_to_dir(selector, dump_dir)
    selector_loaded = load_selector_from_dir(dump_dir)
    x2 = selector_loaded.transform(x_all).astype(np.float64)
    assert x.shape == x2.shape
    assert x.dtype == x2.dtype
    assert (x != x2).nnz == 0

def test_diff_version(caplog):
    print(sklearn.__version__)
    if sklearn.__version__ == "0.21.3":
        assert True
    else:
        vectorizer_dir = "./tests/artefacts/0"
        vectorizer_loaded = load_vectorizer_from_dir(vectorizer_dir)
        assert caplog.records[-1].levelname == "WARNING"
        assert "The version of sklearn" in  caplog.records[-1].msg
