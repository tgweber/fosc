################################################################################
# Copyright: Tobias Weber 2020
#
# Apache 2.0 License
#
# This file contains code related to the Field of Study Classification
#
################################################################################

import json
import logging
import numpy as np
import os
import pickle
import sklearn
import tarfile
import urllib.request
import warnings

__version__ = "0.0.1"

# We silence tensorflow output before we load it
# https://stackoverflow.com/a/54950981
def _tf_shutup():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
_tf_shutup()
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from keras.preprocessing.sequence import pad_sequences

from fosc.config import config

def load_model(model_id):
    """ Loads the model. Downloads the model if not present on local drive

    Arguments
    ---------
    model_id: str
        One of mlp_s, mlp_m, mlp_l, lstm_s, lstm_m, lstm_m
        Consult README.md for details on the models

    Returns
    -------
    tensorflow.python.keras.engine.sequential.Sequential for type mlp
    tensorflow.python.keras.engine.training.Model for type lstm
    """
    _download(model_id)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        with open(os.path.join(config["base_path"], model_id, "model.json"), "r") as f:
            model_config = json.load(f)
        model = tf.keras.models.model_from_json(json.dumps(model_config))
        model.load_weights(os.path.join(config["base_path"], model_id, "model_weights.h5"))
    return model

def vectorize(payload, model_id):
    """ Vectorizes texts for usage with the specified model
        Downloads model and serialized python code, if not present local.

    Arguments
    ---------
    texts: pandas Series of str
        Payloads to be vectorized.
    model_id: str
        One of mlp_s, mlp_m, mlp_l, lstm_s, lstm_m, lstm_m
        Consult README.md for details on the models
    Returns
    -------
    something (TBA)
    """
    _download(model_id)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        if config["models"][model_id]["type"] == "mlp":
            with open(os.path.join(config["base_path"], model_id, "vectorizer.bin"), "rb") as f:
                vectorizer = pickle.load(f)
            with open(os.path.join(config["base_path"], model_id,"selector.bin"), "rb") as f:
                selector = pickle.load(f)
            return selector.transform(vectorizer.transform(payload)).astype(np.float64)
        elif config["models"][model_id]["type"] == "lstm":
            with open(os.path.join(config["base_path"], model_id, "tokenizer.bin"), "rb") as f:
                tokenizer = pickle.load(f)
            emb_matrix = np.load(os.path.join(config["base_path"], model_id, "emb_matrix.npy"))
            return pad_sequences(
                tokenizer.texts_to_sequences(payload),
                maxlen=config["models"][model_id]["maxlen"]
        )

def _download(model_id):
    """ Downloads model, weights and vectorization objects

    Argument
    --------
    model_id: str
        One of mlp_s, mlp_m, mlp_l, lstm_s, lstm_m, lstm_m
        Consult README.md for details on the models
    """
    if not os.path.isdir(os.path.join(config["base_path"], model_id)):
        target_path_archive = os.path.join(config["base_path"], model_id + ".tar.gz")
        print("Downloading model {} (might take some time)".format(model_id))
        urllib.request.urlretrieve(config["models"][model_id]["url"], target_path_archive)
        print("Extracting model {} (might take some time)".format(model_id))
        tar = tarfile.open(target_path_archive, "r:gz")
        tar.extractall(path=config["base_path"])
        tar.close()
        os.remove(target_path_archive)
