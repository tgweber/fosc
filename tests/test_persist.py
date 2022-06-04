################################################################################
# Copyright: Tobias Weber 2022
#
# Apache 2.0 License
#
# This file contains all persistance-related tests
#
################################################################################
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


from fosc import \
    dump_vectorizer_to_dir,\
    load_vectorizer_from_dir

def test_store_and_load_vectorizer(tmpdir):
    corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?',
    ]
    vectorizer = TfidfVectorizer(stop_words=["This"], norm="l1")
    vectorizer.fit(corpus)

    vectorizer_dump_dir = tmpdir.mkdir("test-vectorizer").realpath()
    dump_vectorizer_to_dir(vectorizer, vectorizer_dump_dir)
    vectorizer_loaded = load_vectorizer_from_dir(vectorizer_dump_dir)
    assert np.array_equal(vectorizer.idf_, vectorizer_loaded.idf_)
    for key, value in vectorizer_loaded.vocabulary_.items():
        assert vectorizer.vocabulary_[key] == value
    for key, value in vectorizer.vocabulary_.items():
        assert vectorizer_loaded.vocabulary_[key] == value
    for key, value in vectorizer_loaded.get_params().items():
        assert vectorizer.get_params()[key] == value
    for key, value in vectorizer.get_params().items():
        assert vectorizer_loaded.get_params()[key] == value
