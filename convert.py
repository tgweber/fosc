
################################################################################
# Copyright: Tobias Weber 2022
#
# Apache 2.0 License
#
# This file contains conversion code to persist selector and vectorizer
# in npz / json format, not in binary (pickle)
#
################################################################################

import logging
import os

from fosc import \
        dump_selector_to_dir,\
        dump_vectorizer_to_dir,\
        get_selector,\
        get_vectorizer

from fosc.config import config

logging.basicConfig(level=logging.INFO)

for model_id in ["mlp_s", "mlp_m", "mlp_l"]:
    logging.info("Handling {}".format(model_id))
    vectorizer = get_vectorizer(model_id)
    dump_vectorizer_dir = os.path.join(
        config["base_path"],
        model_id,
        "vectorizer"
    )
    if os.path.isdir(dump_vectorizer_dir):
        logging.info("{} already converted, doing nothing".format(model_id))
        continue
    else:
        os.makedirs(dump_vectorizer_dir)
        dump_vectorizer_to_dir(vectorizer, dump_vectorizer_dir)

    selector = get_selector(model_id)
    dump_selector_dir = os.path.join(
        config["base_path"],
        model_id,
        "selector"
    )
    if os.path.isdir(dump_selector_dir):
        logging.info("{} already converted, doing nothing")
        continue
    else:
        os.makedirs(dump_selector_dir)
        dump_selector_to_dir(selector, dump_selector_dir)
