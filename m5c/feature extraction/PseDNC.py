#PseDNC  Chen, W., et al., iRNA-Methyl: Identifying N6-methyladenosine sites using pseudo nucleotide composition. Analytical biochemistry, 2015. 490: p. 26-33.
import pandas as pd
import numpy as np
import os
import sys
import itertools

path=""
outputname='PseDNC.csv'
gene_type='DNA'
fill_NA='0'
propertyname=r"physical_chemical_properties_RNA.txt"
#if fill_NA=="1" and gene_type=="RNA":
#    propertyname="PseDNC/physical_chemical_properties_3_RNA_without.txt"
#elif fill_NA=="0" and gene_type=="RNA":
#    propertyname="PseDNC/physical_chemical_properties_3_RNA.txt"
#elif fill_NA=="1" and gene_type=="DNA":
#    propertyname="PseDNC/physical_chemical_properties_3_DNA_without.txt"
#elif fill_NA=="0" and gene_type=="DNA":
#    propertyname="physical_chemical_properties_3_DNA.txt"
#print(propertyname)
phisical_chemical_proporties=pd.read_csv(path+propertyname,header=None,index_col=None)
m6a_sequence=open(r'input file', 'r')
DNC_key=phisical_chemical_proporties.values[:,0]
#print (DNC_key)
#print (len(DNC_key))

if fill_NA=="1":
    DNC_key[21]='NA'
# DNC_key=np.array(['AA','AC','AG','AU','CA','CC','CG','CU','GA','GC','GG','GU','UA','UC','UG','UU'])
DNC_value=phisical_chemical_proporties.values[:,1:]
DNC_value=np.array(DNC_value).T
DNC_value_scale=[[]]*len(DNC_value)
#print (len(DNC_value))
for i in range(len(DNC_value)):
    average_=sum(DNC_value[i]*1.0/len(DNC_value[i]))
    std_=np.std(DNC_value[i],ddof=1)
    DNC_value_scale[i]=[round((e-average_)/std_,2) for e in DNC_value[i]]
    #print (DNC_value_scale)
DNC_value_scale=list(zip(*DNC_value_scale))
#print DNC_value_scale



DNC_len=len(DNC_value_scale)
#print (DNC_len)
m6aseq=[]
for line in m6a_sequence:
    if line.startswith('>'):
        pass
    elif line == '\n':
        line = line.strip("\n")
    else:
        m6aseq.append(line.replace('\n','').replace("\r",''))
w=0.9
Lamda=6
result_value=[]
m6a_len=len(m6aseq[0])
#print m6a_len
m6a_num=len(m6aseq)
for m6a_line_index in range(m6a_num):
    frequency=[0]*len(DNC_key)
    #print len(frequency)
    m6a_DNC_value=[[]]*(m6a_len-1)
    #print m6a_DNC_value
    for m6a_line_doublechar_index in range(m6a_len):
        for DNC_index in range(len(DNC_key)):
            if m6aseq[m6a_line_index][m6a_line_doublechar_index:m6a_line_doublechar_index+2]==DNC_key[DNC_index]:
                #print m6aseq[2][0:2]
                m6a_DNC_value[m6a_line_doublechar_index]=DNC_value_scale[DNC_index]
                frequency[DNC_index]+=1
    #print m6a_DNC_value
    
    frequency=[e/float(sum(frequency)) for e in frequency]
    p=sum((frequency))
    #print p
    #frequency=np.array(frequency)/float(sum(frequency))#(m6a_len-1)
    one_line_value_with = 0.0
    sita = [0] * Lamda
    #print len(sita)
    for lambda_index in range(1, Lamda + 1):
        one_line_value_without_ = 0.0
        for m6a_sequence_value_index in range(1, m6a_len - lambda_index):
            temp = list(map(lambda x,y : round((x - y) ** 2,8), list(np.array(m6a_DNC_value[m6a_sequence_value_index - 1])),list(np.array(m6a_DNC_value[m6a_sequence_value_index - 1 + lambda_index]))))
            #map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
            temp_value = round(sum(temp) * 1.0 / DNC_len,8)
            one_line_value_without_ += temp_value
        one_line_value_without_ = round(one_line_value_without_ / (m6a_len - lambda_index-1),8)
        sita[lambda_index - 1] = one_line_value_without_
        one_line_value_with += one_line_value_without_
    dim = [0] * (len(DNC_key) + Lamda)
    #print len(dim)
    for index in range(1, len(DNC_key) + Lamda+1):
        if index <= len(DNC_key):
            dim[index - 1] = frequency[index - 1] / (1.0 + w * one_line_value_with)
        else:
            dim[index - 1] = w * sita[index - len(DNC_key)-1] / (1.0 + w * one_line_value_with)
        dim[index-1]=round(dim[index-1],8)
    result_value.append(dim)
print(np.array(result_value).shape)
pd.DataFrame(result_value).to_csv(r'save file', header=None, index=None)
m6a_sequence.close()