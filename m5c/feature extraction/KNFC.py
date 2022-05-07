# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:38:41 2017

@author: fcq
        
"""

import itertools
import re
import pandas as pd
import numpy as np

"""
import itertools
@details:
    According to the value of K, generate K nucleotides(k-tuple nucleotides)
    such as: 
        k = 1, k nucleotides are: A, T, C, G
        k = 2, k nucleotides are: AA, AT, AC, AG, TA, TT, TG, TC, GA, GT, GG, GC, CA, CT, CG, CC
        k = 3, k nucleotides are: AAA, AAT, ...
        
"""
def ObtainKnucleotides(k):
    bases = ['A', 'U', 'C', 'G']
    k_bases = []
    k_nucleotides = []
    indexs = [''.join(x) for x in itertools.product('0123', repeat=k)]  #generate the permutation and combination with the format '0123'
    
    for i in range(k):
        k_bases.append(bases)
    
    for index in indexs:
        k_indexs = list(index)
        m = ''
        for k_index in k_indexs:
            m = m + k_bases[k_indexs.index(k_index)][int(k_index)]
        k_nucleotides.append(m)
    
    return k_nucleotides   

"""
import re
@details:
    Cacluate the frequency of k_nucleotides in sequence
    such as: 
        sequence: ATACTTCAT  
        k_nucleotides: AT
        frequency of AT: 2/8
    
    D = [f1, f2, f3, ..., f(4**k)]
@paras:
    sequence must be a string, not a list

"""

def CaculateKnucleotidesfrequency(sequence, k_nucleotides):
    fn = []
    k = len(k_nucleotides[0])
    len_seq = len(sequence)
    #print("len_seq: ", len_seq)

    kbases_seq_count = dict()
    kbases_seq_count = kbases_seq_count.fromkeys(k_nucleotides, 0) 
    kbases_seq_frequency = dict()
    kbases_seq_frequency = kbases_seq_frequency.fromkeys(k_nucleotides, 0)

    for seq_i in range(len(sequence) - k + 1):
        kbase = sequence[seq_i:seq_i + k] 
        kbases_seq_count[kbase] += 1
    for kbase in k_nucleotides:
        kbases_seq_frequency[kbase] = kbases_seq_count[kbase] / (len_seq - k + 1)
    fn = list(kbases_seq_frequency.values())
    #print("fn:\n", fn)
    
    return fn

def ObtainSequenceAndLabels(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    sequences = []
    
    for line in lines:
        if line[0] != '>':
            each_line = line.strip()
            sequences.append(each_line)
    
    return sequences

def generate_fn_file(sequences, k):
    k_nucleotides = ObtainKnucleotides(k)
    len_seqs = len(sequences)
    #df.loc[index_], index_ must not be repeated by other indexes
    #Dfn = pd.HDFStore("Dfn.csv", "w")
    
    i = 0  
    value = np.zeros((len_seqs, 4 ** k))
    fn_sum = np.zeros((len_seqs, 1))
    for sequence in sequences:
        print("i = ", i)
        fn = CaculateKnucleotidesfrequency(sequence, k_nucleotides)
        #df.loc[df.index[sequences.index(sequence)]] = fn
        value[i, :] = fn
        fn_sum[i, 0] = sum(fn)        
        i = i + 1

    df = pd.DataFrame(value, index = np.arange(len_seqs), columns = k_nucleotides)   #build a empty DataFrame
    sum_fn = pd.DataFrame(fn_sum)
    df.to_csv("save file",header=None,index=False) #output file
    #sum_fn.to_csv("sum(k=%s).csv"%(str(k)))
    #print("df:\n", df)
    
import os 
import sys
#sequence_file = sys.argv[1]
if __name__ == '__main__':
    sequences = ObtainSequenceAndLabels('input file')
    #generate_fn_file(sequences, 1)#K is the parameter which could pruduce best performance in pre-experiment
    #generate_fn_file(sequences, 2)#K is the parameter which could pruduce best performance in pre-experiment
    #generate_fn_file(sequences, 3)#K is the parameter which could pruduce best performance in pre-experiment
    generate_fn_file(sequences, 4)#K is the parameter which could pruduce best performance in pre-experiment



   
    
    

