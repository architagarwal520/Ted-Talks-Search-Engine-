import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords 
import math
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import pickle





def lexical_analyse(X):
	# tokenization 
	X=X.lower()
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(X)
	X_list=	nltk.word_tokenize(" ".join(tokens)) 
  
# sw contains the list of stopwords 
	sw = stopwords.words('english')  
  
# remove stop words from string 
	X_list = [w for w in X_list if not w in sw] 

#Stemming of tokens, return list
	ps = PorterStemmer()
	stemmed = []
	for i in X_list:
		stemmed.append(ps.stem(i))
	
	return stemmed


if __name__ == '__main__': 
	data = pd.read_csv('models/transcripts.csv')
	Z_list=[]
	documents=[]
	for i in range(len(data)):
		Y_list=lexical_analyse(data.loc[i,"transcript"])
		documents.append(Y_list)
		Z_list=Z_list+Y_list
	set_Z=set(Z_list)


	with open('z_set.pkl', 'wb') as f:
		pickle.dump(Z_list, f)

	with open('doc.pkl', 'wb') as f:
		pickle.dump(documents, f)