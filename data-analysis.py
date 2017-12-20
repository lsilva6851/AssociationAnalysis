import numpy as np
import pandas as pd
from apriori import *
import sys

data = pd.read_csv('nota-codigo-descricao.csv',
                sep=',',
                header=0,
                low_memory=False)

finalList = []

for group, frame in data.groupby(['NOTA']):

    record = []
    for a in frame.values:
        record.append(a[2])
    if len(record) > 1:
        finalList.append(frozenset(record))

minsup = float(sys.argv[1])
minconf = float(sys.argv[2])

items, rules = runApriori(finalList, minsup, minconf)

arules = []
for rule in rules:
    if float(rule[2]) > 1:
        arules.append([str(rule[0][0][0]), str(rule[0][1][0]), float(rule[1]), float(rule[2])])

ar = pd.DataFrame(arules, columns=['pre', 'post', 'confidence', 'lift'])
ar.to_csv('rules-%s-%s.csv' % (minsup, minconf), ignore_index=True)

print(ar)
