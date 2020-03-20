################################################################################
# Copyright: Tobias Weber 2019
#
# Apache 2.0 License
#
# This file contains all RDP-related tests
#
################################################################################

from fosc import \
        _tf_shutup, \
        get_emb_matrix, \
        get_selector, \
        get_tokenizer, \
        get_vectorizer, \
        load_model, \
        vectorize
from fosc.config import config

import numpy as np
import os
import pandas as pd
from scipy.sparse.csr import csr_matrix
import shutil
import pytest

# This is how you import tensorflow stuff without the annoying output
# Thanks to https://stackoverflow.com/a/54950981
_tf_shutup()
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from tensorflow.python.keras.engine.sequential import Sequential
    from tensorflow.python.keras.engine.training import Model

def test_config():
    assert "models" in config.keys()
    assert "labels" in config.keys()
    assert "base_path" in config.keys()
    for model_id in ("mlp_s", "mlp_m", "mlp_l", "lstm_s", "lstm_m", "lstm_l"):
        assert model_id in config["models"].keys()
        assert "url" in config["models"][model_id].keys()
        assert "id" in config["models"][model_id].keys()
        assert "type" in config["models"][model_id].keys()
        if config["models"][model_id]["type"] == "lstm":
            assert "maxlen" in config["models"][model_id].keys()

def test_load_model():
    should_be_removed_for_coverage = os.path.join(config["base_path"], "mlp_s")
    if os.path.isdir(should_be_removed_for_coverage):
        shutil.rmtree(should_be_removed_for_coverage)
    model = load_model("mlp_s")
    assert isinstance(model, Sequential)
    model = load_model("lstm_s")
    assert isinstance(model, Model)

def test_vectorize():
    texts = pd.Series(["psycholgy thought mind", "algorithm computer performance"])
    vectorized = vectorize(texts, "mlp_s")
    assert isinstance(vectorized, csr_matrix)

    vectorized = vectorize(texts, "lstm_s")
    assert isinstance(vectorized, np.ndarray)

def test_exceptions():
    with pytest.raises(ValueError) as ve:
        vectorizer = get_vectorizer("lstm_s")
        assert str(ve) == "lstm_s has no vectorizer"
    with pytest.raises(ValueError) as ve:
        selector = get_selector("lstm_s")
        assert str(ve) == "lstm_s has no selector"
    with pytest.raises(ValueError) as ve:
        tokenizer = get_tokenizer("mlp_s")
        assert str(ve) == "mlp_s has no tokenizer"
    with pytest.raises(ValueError) as ve:
        emb_matrix = get_emb_matrix("mlp_s")
        assert str(ve) == "mlp_s has no emb_matrix"
