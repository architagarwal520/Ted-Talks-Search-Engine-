import pandas as pd
import numpy as np
import nltk
import math
from models.doc_analysis import lexical_analyse
import pickle

def get_cosine(X):
	X_list=lexical_analyse(X)
	data = pd.read_csv('models/transcripts.csv')
	print("data found")
	similarity=[]
	with open('z_set.pkl', 'rb') as f:
		Z_list= pickle.load(f)
	with open('doc.pkl', 'rb') as f:
		documents = pickle.load(f)
	set_Z=set(Z_list)
	q = []#idf-tdf for query
	idf=pickle.load(open("idf.p", "rb"))
	print("got idf")
	for i in idf.keys():
		if i in X_list:
			q.append((X_list.count(i)/len(set(X_list)))*idf[i])
		else:
			q.append(0)
	mod_q=0
	for i in range(len(q)):
		mod_q+=q[i]**2
	mod_q=mod_q**0.5
	with open('tdf_idf.pkl', 'rb') as f:
		tdf_idf_vector= pickle.load(f)
	print("tdf-idf")
	for tdf_idf in tdf_idf_vector:
		dot_product=0
		tdf_mod=0
		for i in range(len(set_Z)): 
			dot_product+= q[i]*tdf_idf[i] 
			tdf_mod+=tdf_idf[i]**2
		tdf_mod=tdf_mod**0.5
			
		cosine = dot_product/ (mod_q*tdf_mod)
		similarity.append(cosine)
	most_similar= []
	print("got similarity")
	min_talks=5
	print("finding url ")
	while min_talks > 0:
		tmp_index = np.argmax(similarity)
		most_similar.append(tmp_index)
		similarity[tmp_index] = 0
		min_talks -= 1
	url=[]
	for i in most_similar:
		link=data.loc[i,"url"]
		url.append(link)
	print("found url")
	return url



