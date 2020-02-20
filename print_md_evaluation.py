################################################################################
# Copyright: Tobias Weber 2020
#
# Apache 2.0 License
#
# This is a utility script to convert the performance scores in evaluation.csv
# to markdown tables for the README.md
#
################################################################################

import argparse
import pandas as pd

from fosc.config import config
import fosc

def print_md(row):
    print("{}:".format(row["model_id"]))
    print("|Field of Study                             |Recall|Precision|")
    print("|-------------------------------------------|------|---------|")
    print("|all (macro)                                |  {:.2f}|     {:.2f}|".format(
        row["recall_all_macro"],
        row["precision_all_macro"])
    )
    print("|all (micro)                                |  {:.2f}|     {:.2f}|".format(
        row["recall_all_micro"],
        row["precision_all_micro"])
    )
    for i in range(0,20):
        print("|{:43}|  {:.2f}|     {:.2f}|".format(
            config["labels"][i],
            round(row["recall_"+str(i)], 2),
            round(row["precision_"+str(i)], 2))
        )
    print("")

parser = argparse.ArgumentParser(
    description='Print the evaluation results for all models in markdown format'
)
parser.add_argument('--version',
    default=str(fosc.__version__),
    help = "Version for which the evaluation result should be printed"
)
args = parser.parse_args()

df = pd.read_csv("evaluation.csv")
df[df.version == args.version].apply(print_md, axis=1)
