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


#OUTPUT
#{'Gensim': 0, 'Python': 1, 'a': 2, 'algorithms.': 3, 'and': 4, 'as': 5, 'designed': 6, 'digital': 7, 'documents': 8, 'efficiently': 9, 'for': 10, 'free': 11, 'is': 12, 'learning': 13, 'library': 14, 'machine': 15, 'open-source': 16, 'painlessly': 17, 'possible.': 18, 'process': 19, 'raw,': 20, 'representing': 21, 'semantic': 22, 'texts': 23, 'to': 24, 'unstructured': 25, 'unsupervised': 26, 'using': 27, 'vectors,': 28}


#section 2

from gensim.utils import simple_preprocess
from gensim import corpora

text2 = open('samplescript.txt', encoding ='utf-8')
 
tokens2 =[]
for line in text2.read().split('.'):
  tokens2.append(simple_preprocess(line, deacc = True))

g_dict2 = corpora.Dictionary(tokens2)

print("The dictionary has: " +str(len(g_dict2)) + " tokens\n")
print(g_dict2.token2id)

# #output
# #{'analyzing': 0, 'and': 1, 'branch': 2, 'consists': 3, 'data': 4, 'deriving': 5, 'efficient': 6, 'for': 7, 'from': 8, 'in': 9, 'information': 10, 'is': 11, 'manner': 12, 'nlp': 13, 'of': 14, 'processes': 15, 'science': 16, 'smart': 17, 'systematic': 18, 'text': 19, 'that': 20, 'the': 21, 'understanding': 22, 'analysis': 23, 'as': 24, 'automated': 25, 'automatic': 26, 'by': 27, 'can': 28, 'chunks': 29, 'components': 30, 'entity': 31, 'etc': 32, 'extraction': 33, 'its': 34, 'machine': 35, 'massive': 36, 'named': 37, 'numerous': 38, 'one': 39, 'organize': 40, 'perform': 41, 'problems': 42, 'range': 43, 'recognition': 44, 'relationship': 45, 'segmentation': 46, 'sentiment': 47, 'solve': 48, 'speech': 49, 'such': 50, 'summarization': 51, 'tasks': 52, 'topic': 53, 'translation': 54, 'utilizing': 55, 'wide': 56}



# #section 3

g_dict2 = corpora.Dictionary(tokens1)

print("The dictionary has: " +str(len(g_dict2)) + " tokens\n")
print(g_dict2.token2id)


# #output-
# # {'Gensim': 0, 'Python': 1, 'a': 2, 'algorithms.': 3, 'and': 4, 'as': 5, 'designed': 6, 'digital': 7, 'documents': 8, 'efficiently': 9, 'for': 10, 'free': 11, 'is': 12, 'learning': 13, 'library': 14, 'machine': 15, 'open-source': 16, 'painlessly': 17, 'possible.': 18, 'process': 19, 'raw,': 20, 'representing': 21, 'semantic': 22, 'texts': 23, 'to': 24, 'unstructured': 25, 'unsupervised': 26, 'using': 27, 'vectors,': 28}
# # The dictionary has: 29 tokens

# # {'Gensim': 0, 'Python': 1, 'a': 2, 'algorithms.': 3, 'and': 4, 'as': 5, 'designed': 6, 'digital': 7, 'documents': 8, 'efficiently': 9, 'for': 10, 'free': 11, 'is': 12, 'learning': 13, 'library': 14, 'machine': 15, 'open-source': 16, 'painlessly': 17, 'possible.': 18, 'process': 19, 'raw,': 20, 'representing': 21, 'semantic': 22, 'texts': 23, 'to': 24, 'unstructured': 25, 'unsupervised': 26, 'using': 27, 'vectors,': 28}



# #Section 4(Bag_of_words)


g_bow =[g_dict1.doc2bow(token, allow_update = True) for token in tokens1]
print("Bag of Words : ", g_bow)

# #output-
# #Bag of Words :  [[(0, 2), (1, 1), (2, 1), (3, 1), (4, 1), (5, 3), (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 2), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1), (19, 1), (20, 1), (21, 1), (22, 1), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1)]]



