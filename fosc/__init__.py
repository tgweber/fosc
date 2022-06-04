################################################################################
# Copyright: Tobias Weber 2020
#
# Apache 2.0 License
#
# This file contains code related to the Field of Study Classification
#
################################################################################

import inspect
import json
import logging
import numpy as np
import os
import pickle
import sklearn
import tarfile
import urllib.request
import warnings

__version__ = "0.0.2"

# We silence tensorflow output before we load it
# https://stackoverflow.com/a/54950981
def _tf_shutup():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('tensorflow').setLevel(logging.FATAL)

_tf_shutup()

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences

from fosc.config import config

class FOSCEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(FOSCEncoder, self).default(obj)


def dump_vectorizer_to_dir(vectorizer, path):
    idf_file = os.path.join(path, "idf.csv")
    np.savetxt(idf_file, vectorizer.idf_, delimiter=",")
    vocabulary_file = os.path.join(path, "vocabulary.json")
    with open(vocabulary_file, "w") as fp:
        json.dump(vectorizer.vocabulary_, fp, cls=FOSCEncoder)
    params_file = os.path.join(path, "params.json")
    non_default_params = {
        "__sklearn_version__": sklearn.__version__
    }
    # get non-default values:
    vectorizer_inspect = \
        inspect.getfullargspec(sklearn.feature_extraction.text.TfidfVectorizer)
    for i, key in enumerate(vectorizer_inspect.args):
        if  i == 0:
            continue
        if vectorizer_inspect.defaults[i-1] != vectorizer.get_params()[key]:
            non_default_params[key] = vectorizer.get_params()[key]
    with open(params_file, "w") as fp:
        json.dump(non_default_params, fp, cls=FOSCEncoder)


def load_vectorizer_from_dir(path):
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
    params_file = os.path.join(path, "params.json")
    with open(params_file, "r") as fp:
        params = json.load(fp)
    sklearn_version = params.pop("__sklearn_version__", "0")
    if sklearn_version != sklearn.__version__:
        logging.warning("The version of sklearn used to create the"
                        " loaded vectorizer {}"
                        " is different from the"
                        " currently used sklearn {}."
                        " Please consult the documentations of both"
                        " versions to determine breaking changes or"
                        " differing default values".format(
                            sklearn_version,
                            sklearn.__version__
                        )
       )
    vectorizer.set_params(**params)
    idf_file = os.path.join(path, "idf.csv")
    vectorizer.idf_ = np.loadtxt(idf_file)
    vocabulary_file = os.path.join(path, "vocabulary.json")
    with open(vocabulary_file, "r") as fp:
        vectorizer.vocabulary_ = json.load(fp)
    return vectorizer


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
            vectorizer = get_vectorizer(model_id)
            selector = get_selector(model_id)
            return selector.transform(vectorizer.transform(payload)).astype(float).sorted_indices()
        elif config["models"][model_id]["type"] == "lstm":
            tokenizer = get_tokenizer(model_id)
            emb_matrix = get_emb_matrix(model_id)
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

def get_vectorizer(model_id):
    """ Returns the vectorizer object for the given model_id

    This function is meant for large classification runs, when the payloads are
    chunked into batches (vectorize()) always loads vectorizer from disk and is
    therefore slow when iteratively called.

    Argument
    --------
    model_id: str
        One of mlp_s, mlp_m, mlp_l
        Consult README.md for details on the models
    """
    _download(model_id)
    if config["models"][model_id]["type"] == "mlp":
        with open(os.path.join(config["base_path"], model_id, "vectorizer.bin"), "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("{} has no vectorizer".format(model_id))

def get_selector(model_id):
    """ Returns the selector object for the given model_id

    This function is meant for large classification runs, when the payloads are
    chunked into batches (vectorize()) always loads selector from disk and is
    therefore slow when iteratively called.

    Argument
    --------
    model_id: str
        One of mlp_s, mlp_m, mlp_l
        Consult README.md for details on the models
    """
    _download(model_id)
    if config["models"][model_id]["type"] == "mlp":
        with open(os.path.join(config["base_path"], model_id, "selector.bin"), "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("{} has no selector".format(model_id))

def get_tokenizer(model_id):
    """ Returns the tokenizer object for the given model_id

    This function is meant for large classification runs, when the payloads are
    chunked into batches (vectorize()) always loads tokenizer from disk and is
    therefore slow when iteratively called.

    Argument
    --------
    model_id: str
        One of lstm_s, lstm_m, lstm_m
        Consult README.md for details on the models
    """
    _download(model_id)
    if config["models"][model_id]["type"] == "lstm":
        with open(os.path.join(config["base_path"], model_id, "tokenizer.bin"), "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError("{} has no tokenizer".format(model_id))

def get_emb_matrix(model_id):
    """ Returns the emb_matrix object for the given model_id

    This function is meant for large classification runs, when the payloads are
    chunked into batches (vectorize()) always loads emb_matrix from disk and is
    therefore slow when iteratively called.

    Argument
    --------
    model_id: str
        One of lstm_s, lstm_m, lstm_m
        Consult README.md for details on the models
    """
    _download(model_id)
    if config["models"][model_id]["type"] == "lstm":
        return np.load(os.path.join(config["base_path"], model_id, "emb_matrix.npy"))
    else:
        raise ValueError("{} has no emb_matrix".format(model_id))
