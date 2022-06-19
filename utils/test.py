import os
import pandas as pd
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from fosc import load_model, vectorize
from fosc.config import config

model_id = 'mlp_s'

df = pd.DataFrame(
    [
        {"id": "Information and Computer Science Example",
         "payload": "Automated classification of metadata of research"
                    " data by their discipline(s) of research can be used"
                    " in scientometric research, by repository service providers,"
                    " and in the context of research data aggregation services."
                    " Openly available metadata of the DataCite index for research"
                    " data were used to compile a large training and evaluation set"
                    " comprised of 609,524 records, which is published alongside this"
                    " paper. These data allow to reproducibly assess classification"
                    " approaches, such as tree-based models and neural networks."
                    " According to our experiments with 20 base classes (multi-label"
                    " classification), multi-layer perceptron models perform best with"
                    " a f1-macro score of 0.760 closely followed by Long Short-Term Memory"
                    " models (f1-macro score of 0.755). A possible application of the"
                    " trained classification models is the quantitative analysis of trends"
                    " towards interdisciplinarity of digital scholarly output or the"
                    " characterization of growth patterns of research data, stratified"
                    " by discipline of research. Both applications perform at scale with"
                    " the proposed models which are available for re-use."
         }
    ]
)
model = load_model(model_id)
vectorized = vectorize(df.payload, model_id)
preds = pd.DataFrame(model.predict(vectorized))
df = df.join(preds)


def pretty_print(row):
    print("{}:".format(row["id"]))
    for i in range(0, 20):
        print("\t{:05.2f}: {}".format(
            round(row[i]*100, 3), config["labels"][i]))


df.apply(pretty_print, axis=1)
