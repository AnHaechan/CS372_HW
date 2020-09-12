# -*- coding: utf-8 -*-
import csv
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from collections import defaultdict

# Overall explanation of my algorithm

#. I dived the problem into 2 subproblems
#   1. How to find modifier-modifiee pairs?
#   2. How to determine uniqueness?

#  1. Find modifier-modifiee pairs

#  1-1. Determining POS of modifier-modifiee pairs
#  Referring from http://schoolsquestiontime.org/what-is-an-adverb/
#  I found that there are 4 types of intensity-modifying pair
#   (1) ADV-ADJ  (deathly sick, deadly serious)
#   (2) ADJ-NOUN (stark contrast, dead center)
#   (3) NOUN-ADJ (pitch black)
#   (4) VERB&ADV (priase highly) 
#   For case (4), VERB and ADV can position in the sentence quite freely.
#   Such as, "I prasied him highly", "I highly praise him",..etc.
#   Therefore, at the first place, I've decided to only treat former 3 cases, (1), (2), (3)
#   Which have relevantly fixed position. 
#
#   In all cases (1),(2) and (3), latter word is modifiee and former is modifier.
#
#   Also, I figured out that
#   (1) and (3) act like ADJ in the sentence
#   (2) acts like NOUN in the sentence
#
#   -----> However also (2) and (3) are both hard to find.
#   For (2), ADJ-NOUN is mostly used not for intensity modifying pair. ADJ just modifies NOUN like 'pretty girl', 
#   as oppose to what we wanted like 'stark contrast'
#   For (3), NOUN-ADJ intensity-modifying idiom is too rare. When we find a NOUN-ADJ in sentence, 
#            mostly NOUN doesn't act as modifier of ADJ. 
#   
#   In summary, I decided to find only case (1) ADV-ADJ  for the sake of correctness and time. 

#  1-2. How to find from the text
#  Followings will be the orders of the process.
#  (1) For each tuple of words (m1,m2) with tag(m1,m2) == (ADV,ADJ) and m1 ends with 'ly'
#      **Why only considering adverb with -ly? : According to http://advancegrammar.blogspot.com/2009/08/types-of-adverb-and-position.html
#        there are 6 types of adverbs : Time, Place, Manner, Frequency, Probability, Degree.
#                                   Ex) before, here, ethically, often, probabily, very
#        What we want is 'unique intensity modifier', so we need to rule out TIme,Place,Frequency,Probability,Degree (they have only limited kinds of words)
#        As adverbs in 'Manner' ends with -ly, I set ADV only ends with -ly.
#  (2) Find the noun that current (ADV-ADJ) pair modifies.
#      **How to find such noun? I set some frequent forms that ADV-ADJ pair modifies noun. That is,
#       1) ADV-ADJ used as modifier
#       1-1) (ADV-ADJ)-NN
#       1-2) (ADV-ADJ)-ADJ-NN
#       1-3) (ADV-ADJ)-CC-ADJ-NN  / CC: and, or,...       
#       2) ADV-ADJ used as predicate
#       NN-BE-(ADV-ADJ) / BE: is, are, were,...
#  (3) Now we have to make sure ADV is used for modifying intensity.
#      If it only changed intensity, the meaning of ADJ and ADV-ADJ should be similar.
#      Check if the meaning is similar or not by 'comparing nouns that they modifies'
#      'The noun' that ADV-ADJ modifies in current sentence vs. 'Nouns' that ADJ soley modifies, found in entire corpus.
#
#  (4) Check path similarity of 'the noun' and each noun in 'nouns' using Wordnet. 
#      If path similarity is large enough, we can ensure that ADV only worked to modifying intensity, not meaning.    
#  
#
#  1-3. How to store the pairs
#  Store using dictionary
#   pair_dict: modifier -> list of modifiee
#  Using default dictionary with [] as default,
#  append whenever finds modifier-modifiee pairs using dict(modifier).append(modifiee)
#
#
#  2. Determining uniqueness
#  The answer of this problem is quite obvious, using the result of problem 1
#  Property 'uniqueness' is bound to modifier, and it is, negative of number of its modifiees
#  uniqueness(modifier) = - len(pair_dict[modifier])
#  
#
#  *. Corpus
#  Brown corpus


print(" Type 'find_restricted_pairs()' to get result. ")

#finding path similarity between to words
def path_similarity_with_words (w1,w2):
    # w1, w2 : string, string / return : float
    #create synset and search path similiarty!
    
    synsets1 = wn.synsets(w1)
    synsets2 = wn.synsets(w2)
    path_similarity = 0  
    
    if (synsets1==None or synsets2==None)or(len(synsets1)==0 or len(synsets2)==0):
        return path_similarity
            
             
    for synset1 in synsets1:
        for synset2 in synsets2:
            new_path_similarity = synset1.path_similarity(synset2)
            if new_path_similarity!=None:
                if new_path_similarity>path_similarity: 
                    path_similarity=new_path_similarity
        
    return path_similarity

#find the noun that ADV-ADJ modifies in the sentence
def corresp_noun1 (rb, jj, tagged_sentence):
    #case1-1
    for i in range(len(tagged_sentence)-2):
        (w1,_) = tagged_sentence[i]
        (w2,_) = tagged_sentence[i+1]
        (w3,t3) = tagged_sentence[i+2]
        
        if w1==rb and w2==jj and t3.startswith('NN'):
            return w3
    #case1-2-1
    for i in range(len(tagged_sentence)-3):
        (w1,_) = tagged_sentence[i]
        (w2,_) = tagged_sentence[i+1]
        (_,t3) = tagged_sentence[i+2]
        (w4,t4) = tagged_sentence[i+3]
        
        if w1==rb and w2==jj and t3.startswith('JJ') and t4.startswith('NN'):
            return w4
    #case1-2-2
    for i in range(len(tagged_sentence)-4):
        (w1,_) = tagged_sentence[i]
        (w2,_) = tagged_sentence[i+1]
        (_,t3) = tagged_sentence[i+2]
        (_,t4) = tagged_sentence[i+3]
        (w5,t5) = tagged_sentence[i+4]
        
        if w1==rb and w2==jj and t3.startswith('CC') and t4.startswith('JJ') and t5.startswith('NN'):
            return w5
    #case2
    for i in range(len(tagged_sentence)-3):
        (w1,t1) = tagged_sentence[i]
        (_,t2) = tagged_sentence[i+1]
        (w3,_) = tagged_sentence[i+2]
        (w4,_) = tagged_sentence[i+3]
        
        if t1.startswith('NN') and t2.startswith('BE') and w3==rb and w4==jj:
            return w1
    
    #fail -> any noun in sentence
    for word,tag in tagged_sentence:
        if tag.startswith('NN'): return word
            
    return ''

# find the noun that NN-ADJ modifies in the sentence
# NOT USED
#def corresp_noun2 (nn, jj, tagged_sentence):
#    for i in range(len(tagged_sentence)-2):
#        (w1, _) = tagged_sentence[i]
#        (w2, _) = tagged_sentence[i+1]
#        (w3, t3) = tagged_sentence[i+2]
#        if w1==nn and w2==jj and t3.startswith('NN'):
#            return w3
#    
#    for i in range(len(tagged_sentence)-3):
#        (w1, t1) = tagged_sentence[i]
#        (_, t2) = tagged_sentence[i+1]
#        (w3, _) = tagged_sentence[i+2]
#        (w4, _) = tagged_sentence[i+3]
#        
#        if t1.startswith('NN') and t2.startswith('BE') and w3==nn and w4==jj:
#            return w1
#        
#    return ''

# find the noun that ADJ modifies in the sentence
def corresp_noun3 (jj, tagged_sentence):
    #case1-1
    for i in range(len(tagged_sentence)-2):
        (w1,_) = tagged_sentence[i]
        (w2,t2) = tagged_sentence[i+2]
        
        if w1==jj and t2.startswith('NN'):
            return w2
    #case1-2-1
    for i in range(len(tagged_sentence)-3):
        (w1,_) = tagged_sentence[i]
        (_,t2) = tagged_sentence[i+1]
        (w3,t3) = tagged_sentence[i+2]
        
        if w1==jj and t2.startswith('JJ') and t3.startswith('NN'):
            return w3
    #case1-2-2
    for i in range(len(tagged_sentence)-4):
        (w2,_) = tagged_sentence[i]
        (_,t3) = tagged_sentence[i+1]
        (_,t4) = tagged_sentence[i+2]
        (w5,t5) = tagged_sentence[i+3]
        
        if w2==jj and t3.startswith('CC') and t4.startswith('JJ') and t5.startswith('NN'):
            return w5
    #case2
    for i in range(len(tagged_sentence)-3):
        (w1,t1) = tagged_sentence[i]
        (_,t2) = tagged_sentence[i+1]
        (w3,_) = tagged_sentence[i+2]
        
        if t1.startswith('NN') and t2.startswith('BE') and w3==jj:
            return w1
    
    #fail 
    return ''

# find the nouns that modifiee(=ADJ) of (ADV,ADJ) pair modifies in the whole corpus
def corresp_nouns_without_modifier(modifiee, tagged_sentences):
    nouns = []
    
    for tagged_sentence in tagged_sentences:
        mynoun = corresp_noun3(modifiee, tagged_sentence)
        if mynoun:
            nouns.append(mynoun)
        
    return nouns

# main function
def find_restricted_pairs ():
    #text & tagged text , dictionary for modifier-modifiee pair
    brown_tagged_text = brown.tagged_words(categories='news') 
    brown_tagged_text=brown_tagged_text
    
    brown_tagged_sents = brown.tagged_sents()
    
    brown_tagged_sents2 = brown.tagged_sents() #used for iteration issue
    
    pair_dict = defaultdict(lambda:[])
   
    print('total num of sents:', len(brown_tagged_sents))
    
    

    for index, tagged_sentence in enumerate(brown_tagged_sents):
        if index%100==0: print(index)
        
        for i in range(len(tagged_sentence)-1):
            (modifier, tag_modifier) = tagged_sentence[i]
            (modifiee, tag_modifiee) = tagged_sentence[i+1]
            
            if tag_modifier.startswith('RB') and tag_modifiee.startswith('JJ') and modifier[-2:]=='ly' and modifiee.islower():
            
                noun = corresp_noun1(modifier, modifiee, tagged_sentence) 
                if not noun: continue
            
                other_nouns = corresp_nouns_without_modifier(modifiee, brown_tagged_sents2) 
                if not other_nouns: continue
                
                is_intensity_modifying_modifier = False        
                for other_noun in other_nouns:
                    if (path_similarity_with_words(other_noun,noun) > 0.32):
                        is_intensity_modifying_modifier=True
                        break
            
                if is_intensity_modifying_modifier: 
                    pair_dict[modifier].append(modifiee)    
        
# for (NN-ADJ) pair
# NOT USED
#            elif (tag_modifier in ['NN']) and tag_modifiee.startswith('JJ'):
#                
#                noun = corresp_noun2(modifier,modifiee, tagged_sentence)
#                other_nouns = corresp_nouns_without_modifier(modifiee, brown_tagged_sents2)
#                is_intensity_modifying_modifier = False
#                
#                for other_noun in other_nouns:
#                    if (path_similarity_with_words(other_noun,noun) > 0.5):
#                        is_intensity_modifying_modifier=True
#                        break
#        
#                if is_intensity_modifying_modifier: 
#                    pair_dict[modifier].append(modifiee)
            
                    
    #Remove redundant modifiees
    pair_dict2 = {key: list(set(pair_dict[key])) for key in pair_dict.keys()}
    
     
    #uniqueness : -len(pair_dict2[modifier])
    pairs = sorted(pair_dict2.items(), key=lambda item:len(item[1]))
    
    f = open('CS372_HW2_output_20180368.csv', 'w')
    for modifier, modifiees in pairs[:100]:
        for modifiee in modifiees:
            pair_str = modifier + ', ' + modifiee + '\n'
            f.write(pair_str)
            
    return pairs[:100]
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




