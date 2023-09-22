#section 1

import gensim
from gensim import corpora

text1 = ["""Gensim is a free open-source Python library for representing documents as semantic vectors,
           as efficiently and painlessly as possible. Gensim is designed 
           to process raw, unstructured digital texts using unsupervised machine learning algorithms."""]

tokens1 = [[item for item in line.split()] for line in text1]
g_dict1 = corpora.Dictionary(tokens1)

print("The dictionary has: " +str(len(g_dict1)) + " tokens\n")
print(g_dict1.token2id)

#section 2

# from gensim.utils import simple_preprocess
# from gensim import corpora

# text2 = open('samplescript.txt', encoding ='utf-8')
 
# tokens2 =[]
# for line in text2.read().split('.'):
#   tokens2.append(simple_preprocess(line, deacc = True))

# g_dict2 = corpora.Dictionary(tokens2)

# print("The dictionary has: " +str(len(g_dict2)) + " tokens\n")
# print(g_dict2.token2id)


#section 3

g_dict2 = corpora.Dictionary(tokens1)

print("The dictionary has: " +str(len(g_dict2)) + " tokens\n")
print(g_dict2.token2id)


#Bag_of_words


g_bow =[g_dict1.doc2bow(token, allow_update = True) for token in tokens1]
print("Bag of Words : ", g_bow)



