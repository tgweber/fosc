# Field Of Study Classification (FOSC)

This is a python module to classify texts about research data products according to their field of study. 
Typical input for the classifier is the concatenated title, abstract and keywords of a 
research data product.

The classification supports 20 classes:
* Mathematical Sciences,
* Physical Sciences,
* Chemical Sciences,
* Earth and Environmental Sciences,
* Biological Sciences,
* Agricultural and Veterinary Sciences,
* Information and Computing Sciences,
* Engineering and Technology,
* Medical and Health Sciences,
* Built Environment and Design,
* Education,
* Economics,
* Commerce, Management, Tourism and Services,
* Studies in Human Society,
* Psychology and Cognitive Sciences,
* Law and Legal Studies,
* Studies in Creative Arts and Writing,
* Language, Communication and Culture,
* History and Archaeology,
* Philosophy and Religious Studies

These classes are very similar to the [ANZSRC (Australian and New Zealand Standard Research Classification)](https://www.abs.gov.au/ausstats/abs@.nsf/0/6BB427AB9696C225CA2574180004463E) classification,
with two exceptions (see paper linked in [models section](#models) for details.

The targed domain of this module is scientometric research (see [models section](#models) for details).

# Quick Start

Especially step 2 takes a lot of time (for just one classification).
The main reason for this is the need to download the models and load models, weights, and vectorizer-objects into the RAM.

Please make sure, that your machine is equipped with sufficient resources to load the model (see [models section](#models) for requirements of each model.

Please be aware that by using this module, serialized code is loaded from zenodo and executed on
your computer.
Make sure you do not run the programs as root and that you only use this repository.

1. Prepare:
```bash
virtualenv -p `which python3` venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run classification:
```python
import pandas as pd
from fosc import load_model, vectorize
from fosc.config import config

model_id = "mlp_s"

df = pd.DataFrame(
    [
        { "id": "Information and Computer Science Example",
          "payload": "Automated classification of metadata of research" \
                     " data by their discipline(s) of research can be used" \
                     " in scientometric research, by repository service providers," \
                     " and in the context of research data aggregation services." \
                     " Openly available metadata of the DataCite index for research"\
                     " data were used to compile a large training and evaluation set" \
                     " comprised of 609,524 records, which is published alongside this" \
                     " paper. These data allow to reproducibly assess classification" \
                     " approaches, such as tree-based models and neural networks." \
                     " According to our experiments with 20 base classes (multi-label" \
                     " classification), multi-layer perceptron models perform best with" \
                     " a f1-macro score of 0.760 closely followed by Long Short-Term Memory" \
                     " models (f1-macro score of 0.755). A possible application of the" \
                     " trained classification models is the quantitative analysis of trends" \
                     " towards interdisciplinarity of digital scholarly output or the" \
                     " characterization of growth patterns of research data, stratified" \
                     " by discipline of research. Both applications perform at scale with" \
                     " the proposed models which are available for re-use."
         } 
    ]
)
model = load_model(model_id)
vectorized = vectorize(df.payload, model_id)
preds = pd.DataFrame(model.predict(vectorized))
df = df.join(preds)```
df.to_csv('predictions.csv')
```

3. Use

This is only to showcase (for small sample only)
```python
def pretty_print(row):
    print("{}:".format(row["id"]))
    for i in range (0,20):
        print("\t{:05.2f}: {}".format(round(row[i]*100,3), config["labels"][i]))
df.apply(pretty_print, axis=1)
```

# Models
The creation and evaluation procedure for the models are explained in a [paper](https://arxiv.org/abs/1910.09313).

The models are downloaded from [zenodo](https://zenodo.org):
* ["mlp_s"](https://doi.org/10.5281/zenodo.3676490) is a small Multi-Layer Perceptron model 
* ["mlp_m"](https://doi.org/10.5281/zenodo.3677336) is a medium Multi-Layer Perceptron model 
* ["mlp_l"](tba) is a large Multi-Layer Perceptron model 
* ["lstm_s"](https://doi.org/10.5281/zenodo.3677342) is a small Long-Short-Term-Memory model 
* ["lstm_m"](https://doi.org/10.5281/zenodo.3677488) is a medium Long-Short-Term-Memory model
* ["lstm_l"](tba) is a large Long-Short-Term-Memory model

The following sections gives information what error is to be expected from the classification and how much RAM must be available for each model.

All perfomance scores are rounded to two decimal positions. They have been calculated with the help of an evaluation set that was not part of the training set and consists of 61,359 records. They can therefore be used to estimate the error, when a model is used to make claims about a distribution:
* _recall_ is the probability that all records of a class class have been classified so by the model. This value can be used to estimate the error of missed classifications (false negative) by multiplying (1-recall) with the number of classified records. Adding this error from the number of predicted records is the upper bound of the predicted number of records.
* _precision_ is the probabiliy that all classifications of the model are correct (for a class). This value can be used to estimate the error of false classifications (false positive) by multiplying (1-precision) with the number of classified records. Subtracting this error from the number of predicted records is the lower bound of the predicted number of records.

Example: 20 records out of a set of texts have been classified with the model "mlp_s" as "Mathematical Science". The estimated (rounded) error range would be 17 to 27 records.

## Small MLP
Small means that 20.000 words from the vocabulary are used for the classification.

### Purpose
This model is not really meant for production usage, it is the smallest version available to test code etc.

### Requirements
Approximately 2.7 GB of free RAM are necessary (for model, weights, and vectorizing objects). It should run on current hardware (if not, ask for a new computer).

### Performance
 
|Field of Study                             |Recall|Precision|
|-------------------------------------------|------|---------|
|all (macro)                                |  0.65|     0.83|
|all (micro)                                |  0.81|     0.85|
|Mathematical Sciences                      |  0.68|     0.77|
|Physical Sciences                          |  0.92|     0.96|
|Chemical Sciences                          |  0.79|     0.80|
|Earth and Environmental Sciences           |  0.72|     0.81|
|Biological Sciences                        |  0.90|     0.87|
|Agricultural and Veterinary Sciences       |  0.41|     0.80|
|Information and Computing Sciences         |  0.75|     0.81|
|Engineering and Technology                 |  0.66|     0.80|
|Medical and Health Sciences                |  0.80|     0.83|
|Built Environment and Design               |  0.44|     0.82|
|Education                                  |  0.57|     0.82|
|Economics                                  |  0.57|     0.79|
|Commerce, Management, Tourism and Services |  0.54|     0.74|
|Studies in Human Society                   |  0.64|     0.79|
|Psychology and Cognitive Sciences          |  0.75|     0.86|
|Law and Legal Studies                      |  0.61|     0.97|
|Studies in Creative Arts and Writing       |  0.53|     0.88|
|Language, Communication and Culture        |  0.58|     0.85|
|History and Archaeology                    |  0.66|     0.83|
|Philosophy and Religious Studies           |  0.45|     0.84|

## Medium MLP 
Medium means that 50.000 words from the vocabulary are used for the classification.

### Purpose
This model can be used for machines with restricted resources, if feasible, consider using the large mlp.

### Requirements
Approximately 3.2 GB of free RAM are necessary (for model, weights, and vectorizing objects).

### Performance

|Field of Study                             |Recall|Precision|
|-------------------------------------------|------|---------|
|all (macro)                                |  0.70|     0.82|
|all (micro)                                |  0.82|     0.87|
|Mathematical Sciences                      |  0.71|     0.80|
|Physical Sciences                          |  0.93|     0.96|
|Chemical Sciences                          |  0.80|     0.83|
|Earth and Environmental Sciences           |  0.77|     0.80|
|Biological Sciences                        |  0.89|     0.89|
|Agricultural and Veterinary Sciences       |  0.44|     0.78|
|Information and Computing Sciences         |  0.76|     0.85|
|Engineering and Technology                 |  0.74|     0.78|
|Medical and Health Sciences                |  0.79|     0.86|
|Built Environment and Design               |  0.56|     0.80|
|Education                                  |  0.68|     0.81|
|Economics                                  |  0.60|     0.78|
|Commerce, Management, Tourism and Services |  0.59|     0.74|
|Studies in Human Society                   |  0.71|     0.77|
|Psychology and Cognitive Sciences          |  0.78|     0.89|
|Law and Legal Studies                      |  0.66|     0.77|
|Studies in Creative Arts and Writing       |  0.62|     0.86|
|Language, Communication and Culture        |  0.70|     0.81|
|History and Archaeology                    |  0.74|     0.83|
|Philosophy and Religious Studies           |  0.54|     0.87|

## Large MLP 
Medium means that 100.000 words from the vocabulary are used for the classification.

### Purpose
This model can be used for the best performance, if resources are restricted, consider
using the medium version or the lstm versions (worse performance, but smaller memory footprint).

### Requirements
Approximately 3.4 GB of free RAM are necessary (for model, weights, and vectorizing objects).

### Performance

|Field of Study                             |Recall|Precision|
|-------------------------------------------|------|---------|
|all (macro)                                |  0.66|     0.85|
|all (micro)                                |  0.83|     0.87|
|Mathematical Sciences                      |  0.72|     0.80|
|Physical Sciences                          |  0.93|     0.97|
|Chemical Sciences                          |  0.82|     0.82|
|Earth and Environmental Sciences           |  0.78|     0.80|
|Biological Sciences                        |  0.90|     0.88|
|Agricultural and Veterinary Sciences       |  0.40|     0.86|
|Information and Computing Sciences         |  0.78|     0.82|
|Engineering and Technology                 |  0.72|     0.79|
|Medical and Health Sciences                |  0.83|     0.85|
|Built Environment and Design               |  0.58|     0.83|
|Education                                  |  0.61|     0.84|
|Economics                                  |  0.60|     0.79|
|Commerce, Management, Tourism and Services |  0.51|     0.80|
|Studies in Human Society                   |  0.63|     0.84|
|Psychology and Cognitive Sciences          |  0.77|     0.87|
|Law and Legal Studies                      |  0.53|     0.93|
|Studies in Creative Arts and Writing       |  0.52|     0.93|
|Language, Communication and Culture        |  0.62|     0.84|
|History and Archaeology                    |  0.61|     0.88|
|Philosophy and Religious Studies           |  0.44|     0.88|

## Small LSTM
Small means that up to the first 500 words of the payload are used for the classification.

### Purpose
This model is not really meant for production usage, it is the smallest version available to test code etc.

### Requirements
Approximately 2.3 GB of free RAM are necessary (for model, weights, and vectorizing objects).

### Performance

|Field of Study                             |Recall|Precision|
|-------------------------------------------|------|---------|
|all (macro)                                |  0.68|     0.81|
|all (micro)                                |  0.81|     0.86|
|Mathematical Sciences                      |  0.69|     0.76|
|Physical Sciences                          |  0.91|     0.97|
|Chemical Sciences                          |  0.76|     0.81|
|Earth and Environmental Sciences           |  0.69|     0.82|
|Biological Sciences                        |  0.89|     0.87|
|Agricultural and Veterinary Sciences       |  0.51|     0.79|
|Information and Computing Sciences         |  0.73|     0.84|
|Engineering and Technology                 |  0.69|     0.81|
|Medical and Health Sciences                |  0.81|     0.85|
|Built Environment and Design               |  0.57|     0.68|
|Education                                  |  0.65|     0.75|
|Economics                                  |  0.61|     0.80|
|Commerce, Management, Tourism and Services |  0.54|     0.74|
|Studies in Human Society                   |  0.73|     0.80|
|Psychology and Cognitive Sciences          |  0.80|     0.83|
|Law and Legal Studies                      |  0.62|     0.90|
|Studies in Creative Arts and Writing       |  0.59|     0.74|
|Language, Communication and Culture        |  0.69|     0.82|
|History and Archaeology                    |  0.69|     0.80|
|Philosophy and Religious Studies           |  0.50|     0.82|

## Medium LSTM
Medium means that up to the first 1000 words of the payload are used for the classification.

### Purpose
This model can be used for machines with restricted resources, if feasible, consider using the large lstm or mlp models. 

### Requirements
Approximately 2.3 GB of free RAM are necessary (for model, weights, and vectorizing objects).

### Performance

|Field of Study                             |Recall|Precision|
|-------------------------------------------|------|---------|
|all (macro)                                |  0.60|     0.82|
|all (micro)                                |  0.76|     0.84|
|Mathematical Sciences                      |  0.58|     0.73|
|Physical Sciences                          |  0.88|     0.97|
|Chemical Sciences                          |  0.60|     0.85|
|Earth and Environmental Sciences           |  0.60|     0.78|
|Biological Sciences                        |  0.93|     0.81|
|Agricultural and Veterinary Sciences       |  0.31|     0.84|
|Information and Computing Sciences         |  0.64|     0.81|
|Engineering and Technology                 |  0.61|     0.81|
|Medical and Health Sciences                |  0.74|     0.83|
|Built Environment and Design               |  0.38|     0.79|
|Education                                  |  0.59|     0.79|
|Economics                                  |  0.51|     0.76|
|Commerce, Management, Tourism and Services |  0.40|     0.78|
|Studies in Human Society                   |  0.66|     0.79|
|Psychology and Cognitive Sciences          |  0.71|     0.85|
|Law and Legal Studies                      |  0.66|     0.84|
|Studies in Creative Arts and Writing       |  0.50|     0.86|
|Language, Communication and Culture        |  0.64|     0.83|
|History and Archaeology                    |  0.61|     0.80|
|Philosophy and Religious Studies           |  0.47|     0.88|

## Large LSTM
Large means that up to the first 2000 words of the payload are used for the classification.

### Purpose
This model can be used for machines with restricted resources, if feasible, consider using the mlp models. 

### Requirements
Approximately 2.4 GB of free RAM are necessary (for model, weights, and vectorizing objects).

### Performance

|Field of Study                             |Recall|Precision|
|-------------------------------------------|------|---------|
|all (macro)                                |  0.62|     0.82|
|all (micro)                                |  0.76|     0.85|
|Mathematical Sciences                      |  0.57|     0.76|
|Physical Sciences                          |  0.89|     0.97|
|Chemical Sciences                          |  0.63|     0.84|
|Earth and Environmental Sciences           |  0.60|     0.81|
|Biological Sciences                        |  0.90|     0.84|
|Agricultural and Veterinary Sciences       |  0.43|     0.73|
|Information and Computing Sciences         |  0.64|     0.81|
|Engineering and Technology                 |  0.68|     0.79|
|Medical and Health Sciences                |  0.77|     0.83|
|Built Environment and Design               |  0.48|     0.83|
|Education                                  |  0.63|     0.82|
|Economics                                  |  0.53|     0.80|
|Commerce, Management, Tourism and Services |  0.50|     0.71|
|Studies in Human Society                   |  0.63|     0.83|
|Psychology and Cognitive Sciences          |  0.77|     0.82|
|Law and Legal Studies                      |  0.55|     0.89|
|Studies in Creative Arts and Writing       |  0.54|     0.83|
|Language, Communication and Culture        |  0.62|     0.80|
|History and Archaeology                    |  0.61|     0.83|
|Philosophy and Religious Studies           |  0.47|     0.82|

# Run Test
```bash
make test-setup
make test
```
The tests checks whether a download works (everytime) and loads the s-sized models into RAM.
They therefore need time and sufficient RAM.

# Update Policy
If the models are re-trained on new data, this package will get an update. 

Old models will be kept on zenodo - so using older versions of this package will provide reproducible results.

Updates do not happen on a regular schedule, since this project is currently not backed by an institution, but a private project.

If you want to sponsor computing resources pleas feel free to reach out to the maintainer.

# Contributions and Getting Help
If you face problems, please open an issue with enough information to reproduce the problem.

Gross misclassifications are very interesting for us; they can be included in the next training and evaluation round, which will lead to updated weights. Please include your payload and the expected results in an Issue.

If you want to contribute by suggesting better paramters, please open a pull-request with a json file including the parameter values you can find in column params in [evaluation.csv](evaluation.csv). Include your reasoning, why these parameters may be better. 

If you want to contribute by suggesting other models, please open an Issue with the model you propose and reasons why this model may supersede the current performance stats (or reduce the memory footprint while keeping the performance comparable).

Please do not ask for support for scikit-learn or tensorflow installations, please consult the web instead. 
