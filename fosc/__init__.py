###############################################################################
# Copyright: Tobias Weber 2020
#
# Apache 2.0 License
#
# This file contains code related to the Field of Study Classification
#
###############################################################################

import gzip
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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest

__version__ = "0.0.3"

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

IDF_FILENAME = "idf.npz"
VOCABULARY_FILENAME = "vocabulary.json.gz"
PARAMS_FILENAME = "params.json.gz"
PVALUE_FILENAME = "pvalues.npz"
SCORES_FILENAME = "scores.npz"

def dump_dict_to_compressed_json(d, path):
    payload = gzip.compress(json.dumps(d, cls=FOSCEncoder).encode("utf-8"))
    with open(path, "wb") as fp:
        fp.write(payload)

def load_dict_from_compressed_json(path):
    with gzip.open(path) as fp:
        return json.loads(fp.read().decode("utf-8"))

def get_params_with_non_default_values(obj):
    non_default_params = {
        "__sklearn_version__": sklearn.__version__
    }
    cls_inspect = inspect.getfullargspec(type(obj))
    # inspect changed the output of getfullargsspec somewhere.
    # We define the new output format as standard and try the old one
    # if nothing can be found.
    args_index = 4
    default_index = 5
    access_by_key = True
    if len(cls_inspect[args_index]) == 0:
        args_index = 0
        default_index = 3
        access_by_key = False
    for i, key in enumerate(cls_inspect[args_index]):
        if  i == 0:
            continue
        if access_by_key:
            default_value = cls_inspect[default_index][key]
        else:
            default_value = cls_inspect[default_index][i-1]
        if default_value != obj.get_params()[key]:
            non_default_params[key] = obj.get_params()[key]
    return non_default_params

def dump_vectorizer_to_dir(vectorizer, path):
    idf_file = os.path.join(path, IDF_FILENAME)
    np.savez_compressed(idf_file, vectorizer.idf_)
    vocabulary_file = os.path.join(path, VOCABULARY_FILENAME)
    dump_dict_to_compressed_json(vectorizer.vocabulary_, vocabulary_file)
    params_with_non_default_values = get_params_with_non_default_values(
        vectorizer
    )
    params_file = os.path.join(path, PARAMS_FILENAME)
    dump_dict_to_compressed_json(params_with_non_default_values, params_file)


def load_vectorizer_from_dir(path):
    vectorizer = TfidfVectorizer()
    params_file = os.path.join(path, PARAMS_FILENAME)
    params = load_dict_from_compressed_json(params_file)
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
    idf_file = os.path.join(path, IDF_FILENAME)
    vectorizer.idf_ = np.load(idf_file, allow_pickle=False)["arr_0"]
    vocabulary_file = os.path.join(path, VOCABULARY_FILENAME)
    vectorizer.vocabulary_ = load_dict_from_compressed_json(vocabulary_file)
    return vectorizer

def dump_selector_to_dir(selector, path):
    scores_file = os.path.join(path, SCORES_FILENAME)
    np.savez_compressed(scores_file, selector.scores_)
    pvalues_file = os.path.join(path, PVALUE_FILENAME)
    np.savez_compressed(pvalues_file, selector.pvalues_)
    params_with_non_default_values = get_params_with_non_default_values(
        selector
    )
    params_with_non_default_values["k"] = selector.k
    params_file = os.path.join(path, PARAMS_FILENAME)
    dump_dict_to_compressed_json(params_with_non_default_values, params_file)


def load_selector_from_dir(path):
    params_file = os.path.join(path, PARAMS_FILENAME)
    params = load_dict_from_compressed_json(params_file)
    selector = SelectKBest(k=params["k"])
    sklearn_version = params.pop("__sklearn_version__", "0")
    if sklearn_version != sklearn.__version__:
        logging.warning("The version of sklearn used to create the"
                        " loaded selector {}"
                        " is different from the"
                        " currently used sklearn {}."
                        " Please consult the documentations of both"
                        " versions to determine breaking changes or"
                        " differing default values".format(
                            sklearn_version,
                            sklearn.__version__
                        )
       )
    selector.set_params(**params)
    scores_file = os.path.join(path, SCORES_FILENAME)
    selector.scores_ = np.load(scores_file, allow_pickle=False)["arr_0"]
    pvalues_file = os.path.join(path, PVALUE_FILENAME)
    selector.pvalues_ = np.load(pvalues_file, allow_pickle=False)["arr_0"]
    return selector

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
        vectorizer_dir = os.path.join(config["base_path"], model_id, "vectorizer")
        if os.path.isdir(vectorizer_dir):
            return load_vectorizer_from_dir(vectorizer_dir)
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
        selector_dir = os.path.join(config["base_path"], model_id, "selector")
        if os.path.isdir(selector_dir):
            return load_selector_from_dir(selector_dir)
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
