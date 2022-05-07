#kmer
import re
import itertools
from collections import Counter
import numpy as np
import pandas as pd
def kmerArray(sequence, k):
    kmer = []
    for i in range(len(sequence) - k + 1):
        kmer.append(sequence[i:i + k])
    return kmer


def Kmer(fastas, k=2, type="RNA_features", upto=False, normalize=True, **kw):
    encoding = []
    header = ['#', 'label']
    NA = 'ACGU'
    if type in ("RNA_features", 'DNA'):
        NA = 'ACGU'
    else:
        NA = 'ACDEFGHIKLMNPQRSTVWY'

    if k < 1:
        print('Error: the k-mer value should larger than 0.')
        return 0

    if upto == True:
        for tmpK in range(1, k + 1):
            for kmer in itertools.product(NA, repeat=tmpK):
                header.append(''.join(kmer))
        encoding.append(header)
        for i in fastas:
            name, sequence, label = i[0], re.sub('-', '', i[1]), i[2]
            count = Counter()
            for tmpK in range(1, k + 1):
                kmers = kmerArray(sequence, tmpK)
                count.update(kmers)
                if normalize == True:
                    for key in count:
                        if len(key) == tmpK:
                            count[key] = count[key] / len(kmers)
            code = [name, label]
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    else:
        for kmer in itertools.product(NA, repeat=k):
            header.append(''.join(kmer))

        for i in fastas:
            sequence = i.strip()
            kmers = kmerArray(sequence, k)
            count = Counter()
            count.update(kmers)
            if normalize == True:
                for key in count:
                    count[key] = count[key] / len(kmers)
            code = []
            for j in range(2, len(header)):
                if header[j] in count:
                    code.append(count[header[j]])
                else:
                    code.append(0)
            encoding.append(code)
    # np.savetxt("{}-mer".format(k), encoding)
    pd.DataFrame(encoding).to_csv("{}-mer.csv".format(k), header=None, index=False)
    return np.array(encoding)


import sys

from sklearn.preprocessing import MinMaxScaler

sys.path.extend(["../../", "../", "./"])
import sys, os
import pandas as pd
import numpy as np
import argparse

import argparse
def read_fasta(file):
    f = open(file)
    documents = f.readlines()
    string = ""
    flag = 0
    fea=[]
    for document in documents:
        if document.startswith(">") and flag == 0:
            flag = 1
            continue
        elif document.startswith(">") and flag == 1:
            string=string.upper()
            fea.append(string)
            string = ""
        else:
            string += document
            string = string.strip()
            string=string.replace(" ", "")

    fea.append(string)
    f.close()
    return fea
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fasta', required=True, help="fasta file name")
    args = parser.parse_args()
    print(args)
fasta = read_fasta('Dataset.txt')
print(np.shape(fasta))
feature_name=["Kmer"]
feature={"Kmer":"Kmer(fasta)"}
for i in feature_name:
        eval(feature[i])


if __name__ == '__main__':
    main()