import nltk
import csv
from nltk.corpus import wordnet as wn

##-------------------------- How to execute this code?------------------------
##                            find_pairs(<nltk.Text>)

## E.g.) find_pairs(nltk.Text(brown.words(categories='news')))
##----------------------------------------------------------------------------

adverbs = ['very','highly', 'extremely', 'quite', 'more', 'much', 'too', 'pretty']
be_verbs = ['am','are','was','were','is','be']
# to omit be-verbs later, since be-verb does not have intensity.

# Overall algorithm
# (1) for each adverb in adverbs, find 'contexts' that it is used in the corpus.
# *context : words near the word, in this code, 2 front and 2 behind
#            e.g.) (I) (did) good (thing) (too) -> 4 context words of 'good'

# (2) for each word in vocab (vocabulary of text),
#         for each adverb in adverbs,
#             for each context of adverb in cotexts of adverb,
#                 if word is right in front of OR behind the adverb,
#                     then exectue add_triple3() 
# For example, adverb:very, word: hot in context (I am very hot guy) can be
# the case of calling add_triple3()

# (3) In add_triple3(), first we try to find single word(called modified_word) 
# whose close_context is equal to close_context of (adverb+word) or (word+adverb)
# *close_context: 1 front and 1 behind

# For example, close_context of (adverb+word) 'very hot' :
# " I am 'very hot' guy " --> am, guy
# close_context of (modified_word) 'attractive' :
# " I am 'attractive' guy " --> am, guy

# (4) If found so, filter through Wordnet's path_similarity
# Compute path_similarity of word and modified word
# For example, 'hot' and 'attractive'  
# If one word has several synsets, try for all cases and select the largest similarity
# and compare to 0.5, if higher, then add to triples

# (5) After all iterations, list(set(triples)) to remove redundancy.



# function for finding 'context'.
# *context : words near the word, in this code, 2 front and 2 behind
def find_context(text, goal_word): 
    #input : nltk.Text, string
    #return: list of (quadraple) () () goal_word () () 
    text_list = list(text)
    contexts = []
    for i in range(1,len(text_list)-3):
        if (text_list[i+1] == goal_word):
            contexts.append((text_list[i-1],text_list[i],text_list[i+2],text_list[i+3]))
    return contexts

def add_triples3(triples,desire_context,text_list,adverb,word):
    #input : list of triple, tuple, list of word(string), string, string
    #output: list of triple
      
    
    # In add_triple3(), first we try to find single word(called modified_word) 
    # whose close_context is equal to close_context of (adverb+word) or (word+adverb)
    # *close_context: 1 front and 1 behind
    
    # For example, close_context of (adverb+word) 'very hot' :
    # " I am 'very hot' guy " --> am, guy
    # close_context of (modified_word) 'attractive' :
    # " I am 'attractive' guy " --> am, guy
    
    for i in range(len(text_list)-2):
        if (text_list[i]==desire_context[0] and text_list[i+2]==desire_context[1]):
            modified_word = text_list[i+1]
            
            if (word in be_verbs or modified_word in be_verbs):
                continue
            if (word.lower() == modified_word.lower()):
                continue
            # If found so, filter through Wordnet's path_similarity
            # Compute path_similarity of word and modified word
            # For example, 'hot' and 'attractive'  
            # If one word has several synsets, try for all cases and select the largest similarity
            # and compare to 0.5, if higher, then add to triples
            
            s1 = wn.synsets(word)
            s2 = wn.synsets(modified_word)
            if s1==None or s2==None:
                continue
            if len(s1)==0 or len(s2)==0:
                continue

            path_similarity = 0            
            for syn1 in s1:
                for syn2 in s2:
                    new_path = syn1.path_similarity(syn2)
                    if new_path!=None:
                        if new_path>path_similarity: path_similarity=new_path
            if path_similarity == None:
                continue
            if (path_similarity> 0.5) and word!=modified_word:
                triples.append((modified_word,adverb,word))
                return triples
    return triples


##function we are going to execute
def find_pairs(text): 
    # input: <nltk.Text> 
    # output : list of triple
    
    triples = []
    vocab = sorted(set([word.lower() for word in text if word.isalpha()]))
    text_list = list(text)
    
    # update list of contexts for each adverb  
    adverb_contextlist_dict = dict()
    for adverb in adverbs:
        adverb_contextlist_dict[adverb]=find_context(text, adverb)
   
    
    #   for each word in vocab (vocabulary of text),
    #         for each adverb in adverbs,
    #             for each context of adverb in cotexts of adverb,
    #                 if word is right in front of OR behind the adverb,
    #                     then exectue add_triple3() 
    # For example, adverb:very, word: hot in context (I am very hot guy) can be
    # the case of calling add_triple3()
    # For example, adverb:very, word: hot in context (I am very hot guy) can be
    # the case of calling add_triple3()
    for word in vocab:
        for adverb in adverbs:
            adverb_contextlist= adverb_contextlist_dict[adverb]
           
            for adv_context in adverb_contextlist:
                #1. () adverb word ()
                if word==adv_context[2]:
                    desire_context = (adv_context[1], adv_context[3])
                    triples = add_triples3(triples,desire_context,text_list,adverb,word)    
                #2. () word adverb ()
                if word==adv_context[1]:
                    desire_context = (adv_context[0], adv_context[2])
                    triples = add_triples3(triples, desire_context,text_list, adverb, word)
            

    
    
    # After all iterations, list(set(triples)) to remove redundancy.
    result = list(set(triples))[:50]
    
    f = open('CS372_HW1_output_20180368.csv', 'w')
    for triple in result:
        triple_str = triple[0] + ', ' + triple[1] + ', ' + triple[2] + '\n'
        f.write(triple_str)
    return result
        
    