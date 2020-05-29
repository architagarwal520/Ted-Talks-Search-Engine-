import pandas as pd
import numpy as np
import nltk
import math
import pickle

def get_tdf(word,document):

		raw_frequency = document.count(word)
		if raw_frequency == 0:
			return 0
		else:
			return raw_frequency



def tdf_idf(document,set_Z,idf):
	t_i_vector=[]
	for w in set_Z:
		value=idf[w]*get_tdf(w,document)
		t_i_vector.append(value)
	return t_i_vector    



tdf_idf_vector=[]
with open('z_set.pkl', 'rb') as f:
	Z_list= pickle.load(f)
with open('doc.pkl', 'rb') as f:
	documents = pickle.load(f)
set_Z=set(Z_list)
idf=pickle.load(open("idf.p", "rb"))

for document in documents:
	tdf_idf_vector.append(tdf_idf(document,set_Z,idf))

with open('tdf_idf.pkl', 'wb') as f:
	pickle.dump(tdf_idf_vector, f)