import pandas as pd
import numpy as np
import nltk
import math
import pickle


def docs_contain_word(word,documents):
	counter = 0
	for document in documents:
		if word in document:
			counter=counter+1
	return counter  


def get_IDF(documents,set_Z):
	idf = {}
	for word in set_Z:
		contains_word = docs_contain_word(word,documents)
		idf[word] = math.log(len(documents)/(contains_word))
	return idf


documents=[]
Z_list=[]
with open('z_set.pkl', 'rb') as f:
		Z_list= pickle.load(f)
with open('doc.pkl', 'rb') as f:
		documents = pickle.load(f)

set_Z=set(Z_list)

idf=get_IDF(documents,set_Z)
pickle.dump(idf, open("idf.p", "wb"))
	