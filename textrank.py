import gensim.models
import spacy
import os
import wikipedia
import numpy as np
from gensim.models import KeyedVectors
from urllib.request import urlretrieve
from spacy_lefff import LefffLemmatizer
from spacy.language import Language
from gensim.corpora import Dictionary
from math import sqrt, acos, pi

if 'frw2v.bin' not in os.listdir():
 urlretrieve('https://s3.us-east-2.amazonaws.com/embeddings.net/embeddings/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin','frw2v.bin')
w2v = KeyedVectors.load_word2vec_format("frw2v.bin",binary=True)
"""print(w2v.index_to_key)"""
wikipedia.set_lang('fr')
writers = []
if len(writers)==0:
 writers = ['Corneille',"Racine",'Flaubert','Balzac','Zola','Baudelaire','Rimbaud','Verlaine']
 corpus  = [ wikipedia.summary(auteur) for auteur in writers ]
num_text = len(corpus)
"""print(num_text)"""

writers = ['Corneille',"Racine",'Flaubert','Balzac','Zola','Baudelaire','Rimbaud','Verlaine']
writersVec=dict()
def get_vector_Writer():
 global writersVec, writers
 for auteur in writers:
  writersVec[auteur] = np.random.rand(200)
 return writersVec
if len(writersVec) ==0:
 get_vector_Writer()

nlp_fr = spacy.load("fr_core_news_sm")
nlp_eng = spacy.load("en_core_web_sm")
nlp_eng.vocab["«"].norm_ = '«'

def text_tokenize_sentence(text)->list:
 doc = nlp_eng(text.replace("\r", "").replace("\n",""))
 return [sent.text for sent in doc.sents]

"""Lemmatization pour la langue francaise"""
"""https://github.com/sammous/spacy-lefff"""
@Language.factory('french_lemmatizer')
def create_french_lemmatizer(nlp, name):
    return LefffLemmatizer()

nlp_fr.add_pipe("french_lemmatizer", name='lefff')
"""doc = nlp(u"Paris est une ville très chère.")
for d in doc:
    print(d.text, d.pos_, d._.lefff_lemma, d.tag_, d.lemma_)"""
def sent_tokenize(sentence)->list: # sentence is a string
 doc = nlp_fr(sentence)
 res = []
 for t in doc:
  if t._.lefff_lemma is None:
   continue
  elif t.is_alpha and not (t.is_space or t.is_stop or t.is_punct or t.like_num):
   res.append(t._.lefff_lemma)
 return res

"""calculer une représentation vectorielle de la phrase par sac de mots??"""
"""Tfid_Model ou sum ou moyenne?"""
def vectorize_sent(token_list):
 res = np.array([0.00 for i in range(200)])
 """0.01 pour les phrase inconnus"""
 if token_list:
  for word in token_list:
   if word in w2v.index_to_key:
    res = np.add(res, w2v.get_vector(word))
   else:
    continue
 return res

def make_vector_list(sentence_lst):
 vect_lst = []
 for sentence in sentence_lst:
  tmp = sent_tokenize(sentence)
  if len(tmp) != 0:
   vect_lst.append(vectorize_sent(tmp))
  else:
   tmp = np.array([0.001 for i in range(200)], dtype = np.float32)
   vect_lst.append(tmp)
 return vect_lst

"""print(vectorize_sent(sent_tokenize("mon nom est zola. m'a dit.")))"""
"""print(len(vectorize_sent(sent_tokenize("mon nom est zola. m'a dit."))))"""
"""Troisième exercice : Définir une fonction qui calcule les similarités entre les phrases"""

def get_Stochastic_matrix(matrix)->int:
 res = matrix
 for idx in range(len(matrix)):
  somme = np.sum(matrix[idx], dtype=np.float32)
  for idx2 in range(len(matrix)):
    res[idx][idx2]= (matrix[idx][idx2])/somme
  """print(sum(res[idx]))"""
 return res

def make_matrix(sentence_lst):
 vector_lst = make_vector_list(sentence_lst)
 num_sent = len(vector_lst)
 matrix = np.full((num_sent,num_sent), 0.0001, dtype=np.float32)
 for i in range(num_sent):
  for j in range(i):
   if i!=j:
    matrix[i][j] = matrix[j][i] = get_angular_similarity(vector_lst[i], vector_lst[j])
 return get_Stochastic_matrix(matrix)

def get_angular_similarity(a,b):
 norms = sqrt(np.dot(a, a)) * sqrt(np.dot(b, b))
 angular_similarity = 0
 if norms != 0:
  cos_sim = np.dot(a, b)/ norms  # acos is valid only if entry number is between [-1,1]
  angular_similarity = 1- acos(cos_sim) / pi
 return angular_similarity

def pagerank(G):
 P = np.random.rand(len(G))
 """print(P)"""
 eps = 10
 while(eps>0):
  P = np.dot(G,P)
  eps-=1
 return P
"""lst_sentence = text_tokenize_sentence(corpus[0])
matrix = make_matrix(lst_sentence)"""

def Kmeilleures(lst_sent, K):
 pageR = pagerank(make_matrix(lst_sent))
 sentR = list((lst_sent[i], float(pageR[i])) for i in range(len(pageR)))
 tmp = sorted(sentR, key = lambda x: x[1], reverse=True)
 return [tmp[i][0] for i in range(K)]
"""print(Kmeilleures(lst_sentence, 5))"""
"""print(len(lst_sentence))"""

def summary(text, k):
 return Kmeilleures(text_tokenize_sentence(text), k)

def printSummary(corpus,k):
 for text in corpus:
  """print(text)"""
  summar = summary(text,k)
  print(summar)
  """print(len(summary(text,k)))"""

printSummary(corpus,5)