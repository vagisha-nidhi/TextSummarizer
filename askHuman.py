# -*- coding: utf-8 -*-

from __future__ import print_function
import re
from nltk.corpus import stopwords
import nltk
import collections
import math
from sklearn.feature_extraction.text import TfidfVectorizer
import entity2
import numpy as np
import rbm
import math
from operator import itemgetter
import pandas as pd

# In[17]:

stemmer = nltk.stem.porter.PorterStemmer()
WORD = re.compile(r'\w+')


# In[18]:

caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"


def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + caps + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(caps + "[.]" + caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(caps + "[.]" + caps + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + caps + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

def humanGenerator(text) :
	sentences = split_into_sentences(text)
	print(len(sentences))
	yesOrNo = []
	print("\nThe sentences are : ")
	print(sentences)
	print("Input 1 to include and 0 to exclude : \n" )
	for x in range(len(sentences)):
		print(sentences[x])
		ans = input("1 or 0 : ")
		yesOrNo.append(ans)

	return yesOrNo

def automaticGenerator(indeces_extracted,text_len) :
	autoYesOrNo = []
	for x in range(text_len):
		if x in indeces_extracted:
			autoYesOrNo.append(1)
		else :
			autoYesOrNo.append(0)

	return autoYesOrNo

def compareHumanAndAutomatic(human,auto):
	count_retrieved = 0
	count_relevant = 0
	for x in range(len(human)):
		if(human[x] == 1):
			count_relevant = count_relevant+1
	for x in range(len(auto)):
		if(auto[x] == 1):
			count_retrieved = count_retrieved+1

	count_intersection = 0
	for x in range(len(human)):
		if(human[x] == 1 and auto[x] == 1):
			count_intersection = count_intersection+1

	precision = count_intersection*1.0/count_retrieved
	recall = count_intersection*1.0/count_relevant
	Fscore = 2*precision*recall/(precision+recall)

	return precision,recall,Fscore

def readText():
	file = open('article1', 'r')
	text = file.read()
	humanGenerator(text)


