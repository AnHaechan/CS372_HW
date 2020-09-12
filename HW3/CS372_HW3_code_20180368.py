# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:41:27 2020

@author: haechan
"""

import nltk
from bs4 import BeautifulSoup 
from pickle import dump
from pickle import load
from convokit import Corpus, download

from urllib import request
from nltk.corpus import wordnet as wn
from collections import defaultdict


###Overall Description of my algorthim

##0. Raw data into tagged corpus
#I used comments in r/wordplay of reddit for the corpus, since it has multiple occurrence of heteronyms 
#but not too much. And got raw data using Convokit api.
#Then tokenized and tagged using nltk tokenizer and tagger. 

##1.Extract heteronyms
#At first, I constructed the vocabulary of corpus, which is set of lemmatized(by wordnet Lemmatizer) words 
#appeared in the corpus. (Used lemmatizing in order to reduce crawling process)
#And then crawled the data:(POS, pronunciation) of each word from the Cambridge Online Dictionary. 
#For every word in vocabulary, if there exist two or more pronunciations for it, I marked it as heteronym.
#Since there was no clear criteria of ‘how much pronunciation &meaning should differ to be a heteronym’ 
#in the HW description, I chose to make a word heteronym even if there is only small difference in accent
# and meaning. ‘Different IPA string’ means different pronunciations. 
#I excluded abbreviation.
#After extracting heteronyms, I crawled from the dictionary again to get ‘definitions’ of each heteronym. 
#There exist one or more definitions for each heteronym.

##2.Annotate pronunciations 
#Input is word ,POS and sentence, and Output is pronunciation.
#There is two cases: (1) There exist one or more POS which has two or more pronunciations (2) Otherwise
#For case (2), it is a straightforward process to choose pronunciation. 
#First lemmatize the word and search the word from heteronym dictionary, 
#then choose pronunciation correspond to POS.
#For case (1), since the ML is banned in this HW, I used dictionary-based method for word sense disambiguation 
#using wordnet and lesk. First, find the expected synset of word in the sentence, 
#using lesk(gave tag as input to increase accuracy). 
#And then, there are list of ‘possible pronunciations’ for that word. 
#As I mentioned earlier, we have ‘definitions’ data for each pronunciation. 
#For each pronunciation, for each definition, find the ‘first word in the definition that has the same tag 
#as desired word(heteronym of which pronunciation is what we’re finding)’s tag’.
# I used this trick because generally, in the definition, similar word is usually used in the front. 
#With this first word, get synset of it and compute path similarity with desired word’s synset. 
#Get maximum value of path similarities (from definitions) and set it as ‘path similarity of the pronunciation’. 
#Among all possible pronunciations, choose one with highest similarity.

##3. Ranking
#1. Number of heteronyms
#2. max ‘portion’ of heteronym
#Portion(heteonym) = (number of heteronym) / (number of heteronyms)
#For heteronym in heteronyms, find max portion(heteronym)
#3. negative of (number of ‘POS-unique’ heteronyms)
#Heteronym is ‘POS-unique’ heteronym if there exist only one pronunciation for a POS,







## function that crawls the Cambridge dictionarny
# word : word that will be searched in the dictionary
# isRecur : for dealing internal problem, just use default when calling
# isHetero : We crawl once again, only for heteronyms. In this case, crawl definitions, too.

def crawl_dictionary(word, isRecur=False, isHetero=False):
    if (word == ''): return []
    
    url = "https://dictionary.cambridge.org/dictionary/english/" + word 
            
    try:
        response=request.urlopen(url)
    except:
        return []
    
    html_current= response.read().decode('utf8') ##8?16?
    html_parsed = BeautifulSoup(html_current, 'html.parser')    
    if html_parsed.find('div',{'data-id':'cald4'})==None: return []
   
    
     
    entrybodyel_list = html_parsed.find('div',{'data-id':'cald4'}).find_all('div',{'class':'pr entry-body__el'})
    info_list=[]  
  
    for entrybodyel in entrybodyel_list:
        ####################For one pos&pronunciation (devided by Cambridge Dictionary)
        info = []

        posbody = entrybodyel.find('div',{'class':'pos-body'}) 
        definitions= posbody.find('div',{'class':'sense-body dsense_b'}).find_all('div',{'class':'def-block ddef_block'})
        
        
          ################Check wheter it is abbrevitaion or not
        if not isHetero:
            isAbbr = False
            for definition in definitions:
                span= definition.find('div',{'class':'ddef_h'}).find('div',{'class':'def ddef_d db'}).find('span',{'class':'lab dlab'})
                if span!=None:
                    #print(span.text)
                    if "abbreviation" in span.text or "short" in span.text:
                          isAbbr=True
                          break

            
            if (isAbbr): continue
     
        ############# If not abbr, Pronunciation,  POS (, definition)
           
        #########POS
        posheader = entrybodyel.find('div',{'class':'pos-header dpos-h'})
    
        tmp = posheader.find('div',{'class':'posgram dpos-g hdib lmr-5'})
        if (tmp==None):
            if not isRecur:
                return crawl_dictionary(lemmatize(word,'V'),True,isHetero)
            else:
                return []
        info.append(tmp.text)  
        ##########
        
        ##########Pronunciation
        try:
            info.append(posheader.find('span',{'class':'us dpron-i'}).find('span',{'class':'pron dpron'}).text)
        except:
            return []
        ###########        
        
        ##########If Hetero-> definitions
        if isHetero:
            defs_text = []
            for definition in definitions:
                def_text = definition.find('div',{'class':'ddef_h'}).find('div',{'class':'def ddef_d db'}).text
                defs_text.append(def_text)
            info.append(defs_text)
            

        ###########
            
            
        info_list.append(info)

    return info_list


## After crawled from dictionary, remove suffix and prefix
# Cambridge dictionary stored suffix, prefix, which cannot be word alone, so remove them
# Also there is cases that some unnecessary additional infos, so modify it
def removefix(info):
    newinfo = []
    for elem in info:
        if not elem[0].endswith('fix'):
            newinfo.append(elem)
            
    newinfo2=[]
    for elem in newinfo:
        if not elem[1].startswith('/'):
            newinfo2.append(  [elem[0] , elem[1][elem[1].index('/'):] ])
        else:
            newinfo2.append(elem)

    return newinfo2



# depending on the tag given, lemmatize the word 
def lemmatize(word,tag):
    
    if not word.isalpha(): return ''
    
    wnl = nltk.WordNetLemmatizer()
    
    if tag.startswith('V'): 
        return wnl.lemmatize(word, 'v').lower()
    elif tag.startswith('N'):
        return wnl.lemmatize(word,'n').lower()
    elif tag.startswith('J'):
        return wnl.lemmatize(word,'a').lower()
    elif tag.startswith('R'):
        return wnl.lemmatize(word,'r').lower()
    else:
        return word.lower()


## lemmatize the sentence
def lemmatized_sents(tagged_sents):
   # lemmatize(word,tag) for tagged_sent in tagged_sents for word,tag in tagged_sent
    lem_sents=[]
    for tagged_sent in tagged_sents:
        lem_sent=[]
        for word, tag in tagged_sent:
            lem_sent.append(lemmatize(word,tag))
        lem_sents.append(lem_sent)
    return lem_sents

## lemmatize the word 
def lemmatized_sent(tagged_sent):
   # lemmatize(word,tag) for tagged_sent in tagged_sents for word,tag in tagged_sent
    lem_sent=[]
    for word, tag in tagged_sent:
        lem_sent.append(lemmatize(word,tag))
    return lem_sent




## Find heteronyms from the dictionary
## By my definition, if there is even small difference in pronunciation, consider it as heteroynm.
## For word 'us', the Cambridge dictionary has wrong format,
## So I modified it, in the sense of 'modifying the external dictionary'
# Also for one-word like 'n', 'a', ... Cambridge dict also has them but they're not formal words mostly.
def find_heteros(english_dictionary):
    heteros = []
    for key in english_dictionary.keys():
        values = english_dictionary[key]
        if len(values) > 1:
            test_pronun = values[0][1]
            if not all([elem[1]==test_pronun for elem in values]):
                heteros.append((key,values))
    
    return [hetero for hetero in heteros if hetero[0]!="us" and len(hetero[0])>1] 
    



## Find sentences with heteronyms and Rank by Priority 1 : num of heteronyms
def rank_hetero_sents (tagged_sents, heteros):

    heteros = [elem[0] for elem in heteros]
    
    ranked_sentences=[]
    for tagged_sent in tagged_sents:
        sent_hetero_pos_list = []
        for word,tag in tagged_sent:
            lemma = lemmatize(word,tag)
            if tag=='VBD' or tag=='VBN':
                lemma = word
                a=lemma.split('ed')[0]
                b=lemma.split('d')[0]
                if a in heteros:
                    sent_hetero_pos_list.append((a,tag))
                elif b in heteros:
                    sent_hetero_pos_list.append((b,tag))
                
            if lemma in heteros:
                sent_hetero_pos_list.append((lemma,tag))
        if sent_hetero_pos_list:
            ranked_sentences.append((tagged_sent,sent_hetero_pos_list,len(sent_hetero_pos_list))) 
    return sorted(ranked_sentences, key= lambda elem:elem[2], reverse=True)


## Give 'max portion' of the sentence, used in Priority 2   
def max_portion(annotations):
    freqs = defaultdict(int)
    for word, pronun in annotations:
        freqs[word]+=1
    
    maxval= max(list(freqs.values()))
    return maxval/len(annotations)


## translate POS tag into Cambridge dictionary POS 
def pos_translate(pos):
    
    if pos.startswith('V'): 
        return 'verb'
    elif pos.startswith('N'):
        return 'noun'
    elif pos.startswith('J'):
        return 'adjective'
    elif pos.startswith('R'):
        return 'adverb'
    elif pos.endswith('DT'):
        return 'determiner'
    elif pos.startswith('CC'):
        return 'conjunction'
    elif pos.startswith('IN'):
        return 'preposition'
    elif pos.startswith('PRP'):
        return 'pronoun'
    else:
        return ''

## translate Cambridge POS tag to match with 'lesk' input
def leskpos(tag):
    if tag.startswith('v'): 
        return 'v'
    elif tag.startswith('n'):
        return 'n'
    elif tag.startswith('adjective'):
        return 'a'
    elif tag.startswith('adverb'):
        return 'r'
    else:
        return '' 
    
    
## In the cambridge dictionary,
## there is some cases that multiple pos are binded together
## So separate them,
## and do some additional actions(to match the pos division)
def modify_pos_pronun_defs (hetero_dict, hetero):
 
    pos_and_pronuns = hetero_dict[hetero]
    
    pos_and_pronuns = [[pos.split('[')[0].strip(' ') , pronun, defs] for [pos, pronun, defs] in pos_and_pronuns] #1. 설명제거
    new_pos_and_pronuns = []
    for [pos, pronun,defs] in pos_and_pronuns:
        poss = pos.split(', ')
        if len(poss) >1 :
            for pos in poss:
                if not [pos,pronun,defs] in new_pos_and_pronuns:
                    if pos=='auxiliary verb': pos='verb'
                    if pos=='ordinal number': pos='adjective'
                    new_pos_and_pronuns.append([pos,pronun,defs])
        else :
            if pos=='auxiliary verb': pos='verb'
            if pos=='ordinal number': pos='adjective'
            new_pos_and_pronuns.append([poss[0], pronun, defs])
    
    return pos_and_pronuns 
 #   return new_pos_and_pronuns






## find pronunciation of the heteronym in the sentence
def find_pronun(hetero,hetero_pos,tagged_sent,hetero_dict):

    
    modified_pos_pronun_defs = modify_pos_pronun_defs(hetero_dict, hetero)
    
    hetero_pos = pos_translate(hetero_pos)  
    
    
    # First, find possible pronunciations
    pronuns_possible = []  
    for [pos,pronun,defs] in modified_pos_pronun_defs:
        if pos==hetero_pos:
            pronuns_possible.append([pos, pronun,defs])
        

        
    #1. 'POS-Unique heteronym'
    if len(pronuns_possible)==1:
        return pronuns_possible[0][1]
    #2. Otherwise (difficult case)
    else:
        
        # This is the part I've described earlier, using the synset part.
        
        untagged_sent = " ".join((word for word in lemmatized_sent(tagged_sent)))
        if leskpos(hetero_pos):
            current_synset = nltk.wsd.lesk(untagged_sent, hetero, leskpos(hetero_pos))
        else:
            current_synset = nltk.wsd.lesk(untagged_sent, hetero)
        
        if current_synset is None: 
            if pronuns_possible:
                return pronuns_possible[0][1]
            else:
                return hetero_dict[hetero][0][1]
            
        return_pronun = hetero_dict[hetero][0][1]
        max_val = 0
        
        for pos, pronun,defs in pronuns_possible:
            synsets=[]
            
            for deftxt in defs:
                tagged_txt = nltk.pos_tag(improved_word_tokenize(deftxt))
                for word,word_pos in tagged_txt:
                    new_pos= pos_translate(word_pos)
                    if new_pos==pos:
                        if leskpos(pos):
                            
                            if wn.synsets(word,leskpos(pos)):
                                synsets.append(wn.synsets(word, leskpos(pos))[0])
                        else:
                            if wn.synsets(word):
                                synsets.append(wn.synsets(word)[0])

            currMax =max([current_synset.path_similarity(synset) for synset in synsets])
            if  currMax> max_val:
                return_pronun= pronun
                max_val=currMax
                
            
            
        return return_pronun
        
    
    

    
## annotate the sentences and rank them by priorities
def annotate_and_rank (tagged_sents, heteros, hetero_dict):
    final_list = []
    ranked_sents = rank_hetero_sents(tagged_sents,heteros)
   
    
    for ranked_sent in ranked_sents:
        tagged_sent, hetero_pos_list, len_heteros = ranked_sent
        
        readable_sent = " ".join([word for word,tag in tagged_sent])
        
        hetero_pronuns = [( hetero, find_pronun(hetero,pos,tagged_sent,hetero_dict)  ) for hetero,pos in hetero_pos_list]
        
        citation = "reddit/wordplay"
        
        final_list.append( [ readable_sent, hetero_pronuns , citation ])
        
  
         
    return sorted(final_list, key= lambda x: (len(x[1]) ,max_portion(x[1])), reverse=True )




print("If you're first time, type : 'main_function(True)' ")




def main_function(isNew=False, corpus=[]):
   # 0. tagged_sents 
    tagged_sents = import_process_reddit(corpus,isNew)
   #   1. english dictionary 
    if isNew:
        english_dict = save_corpus(tagged_sents)
    else:
        english_dict = load_corpus()
  #  2. heteronym, heterodict
    heteros = find_heteros(english_dict)
    hetero_dict= get_hetero_dict(heteros, isNew)
    tmp = {key:value for key,value in heteros}
    heteros = [[key,tmp[key]] for key in hetero_dict.keys()]
  #  3.annotate_and_rank
    final_list = annotate_and_rank(tagged_sents,heteros,hetero_dict)  
  #  4. format to csv file
    final_list=final_list[:30]
    f = open('CS372_HW3_output_20180368.csv', 'w' ,encoding='utf-8-sig')
    for sentence, annotations, citation in final_list:
        row=sentence.replace(",", ' ')
        for annotation in annotations:
            annotation = " ".join(annotation)
            row = row + ', ' + annotation 
        f.write(row+', '+citation+'\n')
            
    
    return final_list



## improve nltk. word tokenize
def improved_word_tokenize(sent):
    words =  nltk.word_tokenize(sent)
    new_words=[]
    for word in words:
        if "-" in word[1:-1]:
            new_words+= word.split("-")
        else:
            new_words.append(word)
   
     # There is many ** ** or * * in reddit for stress, just remove them              
    star_splited  = ["".join(word.split("**")) for word in new_words]
    star_splited2  = ["".join(word.split("*")) for word in star_splited]
    result = [word for word in star_splited2 if word!='']
    
    for i in range(len(result)-2):
        if result[i].lower()=="do"  and result[i+1]=="n't":
            result[i]=="don't"
            result.pop(i+1)
    return result

            

## Tokenize and Tag the corpus
def import_process_reddit(corpus, downloadNew = False):
    if downloadNew:
        corpus=Corpus(filename=download("subreddit-wordplay"))
    
    sents = []
    
    utter_ids = corpus.get_utterance_ids()    
    utter_ids = utter_ids[1000:]
    for utter_id in utter_ids:
        sents+= nltk.sent_tokenize(corpus.get_utterance(utter_id).text)
  
    return nltk.pos_tag_sents(improved_word_tokenize(sent) for sent in sents)

    


## First time, crawl the dictionary and save it.
def save_corpus(tagged_sents):
    
    
    english_dictionary = dict()
    
    lemmatized_vocab = set(lemmatize(word,tag) for tagged_sent in tagged_sents for word,tag in tagged_sent )
    
    print(len(lemmatized_vocab))
    for index, word in enumerate(lemmatized_vocab):
        if (index%10==0): print(index)
        info = crawl_dictionary(word)
        
        if info:
            english_dictionary[word] = removefix(info)
 
    
    
    
    output = open('english_dictionary2.pkl', 'wb')
    dump(english_dictionary, output, -1)
    output.close()
    return

## Not first time, just load the dictionary.
def load_corpus():
    
    input = open('english_dictionary2.pkl', 'rb')    
    english_dictionary = load(input)   
    input.close()
    
    
    return english_dictionary


def get_hetero_dict(heteros, isNew=False):
    hetero_dict = dict()
    
    hetero_lemmas = [elem[0] for elem in heteros]
    
    
    if isNew:
        print(len(hetero_lemmas))
        for index, word in enumerate(hetero_lemmas):
            if (index%5==0): print(index)
            info = crawl_dictionary(word,False,True)
        
            if info:
                hetero_dict[word] = removefix(info)
        output = open('hetero_dict.pkl', 'wb')
        dump(hetero_dict, output, -1)
        output.close()
        
    else:
        input = open('hetero_dict.pkl', 'rb')    
        hetero_dict = load(input)   
        input.close()
        
    for key in hetero_dict.keys():
        poslist = hetero_dict[key]
        for enum,elem in enumerate(poslist):
            if len(elem)==2:
                hetero_dict[key][enum].append([key])
                
    return hetero_dict
    
    
main_function(True)
    

#############
    
    






#1. import reddit corpus / 

#2. (reddit corpus)->(raw texts)->(tokenzied sents)->(tagged sents)

#3. for each (word,tag) in sent in sents:
#   (1) lemma = lemmatize_pos(word,tag)
    # lemmatize_pos : tag 따라서 .lemmatize(word,tag2)
      # nltk.pos_tag -> VBD(past tense)인거 잡아냄 
      # WordNet Lemmaitizer -> .lemmatize(word, 'v') // POS설정가능!
#  (2) if isHeteronym(lemma)
    # if lemma in memoizationDict: return memoizationDict['lemma'] // bool
    # else :
    #  result = parser.fetch(lemma)
         ## 검색해서 안나오는경우 (O)
      # (4) find_pronunciations(result) 
       # 1) Ethy 1개     
          # not after "Received Pronunciation", "UK"
          #
       # 2) Ethy 2개 이상
          # rule : merge all text -> 첫 IPA, not after "Received Pronunciation", "UK" 
 
#    (3) if isHeteronym_POS(lemma) // 하나의 POS에 하나의 발음 있을때 
#         return pronun(tag)
# 
#    (3) else: 
#          use lesk? / wn? /    
   
# (2) else: continue
## OUTPUT에 cite (reddit-r/humor) 추가하는거 잊지 말자!!!
    