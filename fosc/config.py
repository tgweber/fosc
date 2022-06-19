###############################################################################
# Copyright: Tobias Weber 2020
#
# Apache 2.0 License
#
# This file contains configurations related to the
# Field of Study Classification
#
###############################################################################
config = {
    "base_path": ".",
    "models": {
        "mlp_s": {
            "id": "mlp_s",
            "type": "mlp",
            "url": "https://zenodo.org/record/6659729/files/mlp_s.tar.gz"
        },
        "mlp_m": {
            "id": "mlp_m",
            "type": "mlp",
            "url": "https://zenodo.org/record/6659758/files/mlp_m.tar.gz"
        },
        "mlp_l": {
            "id": "mlp_l",
            "type": "mlp",
            "url": "https://zenodo.org/record/6659759/files/mlp_l.tar.gz"
        },
        "lstm_s": {
            "id": "lstm_s",
            "type": "lstm",
            "url": "https://zenodo.org/record/6667528/files/lstm_s.tar.gz",
            "maxlen": 500
        },
        "lstm_m": {
            "id": "lstm_m",
            "type": "lstm",
            "url": "https://zenodo.org/record/6667529/files/lstm_m.tar.gz",
            "maxlen": 1000
        },
        "lstm_l": {
            "id": "lstm_l",
            "type": "lstm",
            "url": "https://zenodo.org/record/6667530/files/lstm_l.tar.gz",
            "maxlen": 2000
        }
    },
    "labels": [
        "Mathematical Sciences",
        "Physical Sciences",
        "Chemical Sciences",
        "Earth and Environmental Sciences",
        "Biological Sciences",
        "Agricultural and Veterinary Sciences",
        "Information and Computing Sciences",
        "Engineering and Technology",
        "Medical and Health Sciences",
        "Built Environment and Design",
        "Education",
        "Economics",
        "Commerce, Management, Tourism and Services",
        "Studies in Human Society",
        "Psychology and Cognitive Sciences",
        "Law and Legal Studies",
        "Studies in Creative Arts and Writing",
        "Language, Communication and Culture",
        "History and Archaeology",
        "Philosophy and Religious Studies"
    ]
}
