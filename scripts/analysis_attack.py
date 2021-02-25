import json, os
from openpyxl import Workbook
import string

stringmap = (
    [el for el in string.ascii_uppercase]
    + ["A" + el for el in string.ascii_uppercase]
    + ["B" + el for el in string.ascii_uppercase]
    + ["C" + el for el in string.ascii_uppercase]
)
FNAME = "vurnerable/train"

with open(FNAME, "r") as f:
    data = [json.loads(el) for el in f.readlines()]
wb = Workbook()
ws = wb.active

stat_checker = []
less_than_threshold = []
for idx, item in enumerate(data):
    try:
        context, response, tokenized_response, original_score, scores = (
            item["context"],
            item["response"],
            item["tokenized_response"],
            item["original_nsp"],
            item["vurnerable_nsp"],
        )
    except:
        continue
    stat_checker.extend([el - original_score for el in scores])
    less_than_threshold.extend([el - original_score < -0.01 for el in scores])
from matplotlib import pyplot as plt

plt.hist(stat_checker, bins=50)
plt.savefig("test.png")
print(sum(less_than_threshold) / len(less_than_threshold))
