from nltk import *
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from collections import Counter
import re, math
import collections
import csv

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     #print ("vector1 : ",vec1)
     #print ("vector2 : ",vec2)
     #print (intersection)
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     WORD = re.compile(r'\w+')
     words = WORD.findall(text)
     return Counter(words)

def similar_text(text1, text2):
    max_sim = 0.0
    

    words1=word_tokenize(text1)
    words2=word_tokenize(text2)
    lmtzr = WordNetLemmatizer()

    stop_words = set(stopwords.words('english'))
    #print (stop_words)
    words1=set(words1).difference(stop_words)   #Removing stop words
    words2=set(words2).difference(stop_words)
    #part of speech wordnet deal with noun,adjective,verb,adverb
    nouns1 = [token for token, pos in pos_tag(words1) if pos.startswith('NN')]
    nouns2 = [token for token, pos in pos_tag(words2) if pos.startswith('NN')]
    verbs1 = [token for token, pos in pos_tag(words1) if pos.startswith('VB')]
    verbs2 = [token for token, pos in pos_tag(words2) if pos.startswith('VB')]
    adjectives1 = [token for token, pos in pos_tag(words1) if pos.startswith('JJ')]
    adjectives2 = [token for token, pos in pos_tag(words2) if pos.startswith('JJ')]
    adverbs1 = [token for token, pos in pos_tag(words1) if pos.startswith('RB')]
    adverbs2 = [token for token, pos in pos_tag(words2) if pos.startswith('RB')]
    cardinals1 = [token for token, pos in pos_tag(words1) if pos.startswith('CD')]
    cardinals2 = [token for token, pos in pos_tag(words2) if pos.startswith('CD')]
    #print(nouns1)
    #print(verbs1)
    for w in nouns1:#lemmatize noun
        nouns1.remove(w)
        nouns1.append(lmtzr.lemmatize(w,pos="n"))
    for w in nouns2:
        nouns2.remove(w)
        nouns2.append(lmtzr.lemmatize(w,pos="n"))
    for w in verbs1:#lemmatize verb
        verbs1.remove(w)
        verbs1.append(lmtzr.lemmatize(w,pos="v"))
    for w in verbs2:
        verbs2.remove(w)
        verbs2.append(lmtzr.lemmatize(w,pos="v"))
  
    sum_sim=0.0
    count_words=0
    ws_noun={}
  
    for noun1 in nouns1:        #Word set of nouns (Similar  words)..  (Semantic Similarity)
      max_sim=0.0
      for noun2 in nouns2:
        synsets_1 = wn.synsets(noun1)
        synsets_2 = wn.synsets(noun2)
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
                sim = wn.wup_similarity(synset_1, synset_2)
                if sim is not None:
                    if sim > max_sim:
                        max_sim = sim
      if(max_sim>0.25):           #only include word if sim is greater than 0.25 (threshold)
        ws_noun[noun1]=max_sim
        count_words+=1
        sum_sim+=max_sim
        
    ws_verb={}
    for verb1 in verbs1:        #Word set of similar verbs (Semantic Similarity)
      max_sim=0.0
      for verb2 in verbs2:
        synsets_1 = wn.synsets(verb1)
        synsets_2 = wn.synsets(verb2)
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
                sim = wn.wup_similarity(synset_1, synset_2)
                if sim is not None:
                    if sim > max_sim:
                        max_sim = sim
      if(max_sim>0.25):
        ws_verb[verb1]=max_sim
        count_words+=1
        sum_sim+=max_sim

    #print (ws_noun)
    #print (ws_verb)

    text1 = ' '.join(words1)
    text2 = ' '.join(words2)
    vector1 = text_to_vector(text1)#text to vector
    vector2 = text_to_vector(text2)

    cosine = get_cosine(vector1, vector2)

    #print("Cosine : ",cosine)
    
    if(count_words==0):
         count_words=1
    similarity=cosine+sum_sim/count_words
    #print("sim : ",similarity/2)
    answer=(similarity/2)*100
    answer = (round(answer, 2))/100
    return answer

simm=[]
#Read CSV file (Data set)
#==========================================================
with open('testsmall.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
         qid=row[0]
         q1=row[1]
         q2=row[2]
         sim=similar_text(q1,q2)
         simm.append(sim)
         print(sim)
#==========================================================



