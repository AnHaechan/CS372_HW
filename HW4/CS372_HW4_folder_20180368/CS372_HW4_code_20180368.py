# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:48:22 2020

@author: haechan
"""

import nltk
from nltk.tree import Tree
from nltk.tree import ParentedTree

import pandas as pd

import csv


    

############# Explanation
#The goal of this assignment is convert (sentence) into list of (triple). 
#My methodology handles this problem by following:
# (Raw sentence) ==[nltk word tokenizer]==> (Tokenized sentence) ==[nltk pos tagger]==> 
# (POS-tagged sentence) ==[preprocessing]==> (pre-processed tagged sentence) ==[CFG parsing]==> 
# (parsed tree) ==[relation extraction]==> triples


# 1. tokenizing & pos tagging
#I used nltk.word_tokenize() and nltk.pos_tag() function for two precedent processes.

# 2. preprocessing
#In the preprocessing, I added a number of things to do for compensating the imprecise tagger. 
#There is a tendency that some verbs in 3rd single form are tagged to plural nouns, 
#so I manually changed the tag especially for actions ‘inhibits’, ‘prevents’,’binds’,’induces’. 
#Also I regarded be p.p. by phrase into one verb and ‘bind to’ into one verb since it is phrasal verb. 
#Finally I changed the overall pos tag make it suitable for my grammar. 
#For example, aggregating ‘VBP/Z/D’ into ‘VB’.
#There are many other samll works done.

# 3. CFG parsing

# The grammar is at the down below.
# There are some rules. First, I added coordinate rules for S, NP, VP. AP, PP. 
# For NP, I first separated modifiers to ‘complement’ and ‘adjunct’.
# ‘Adjunct’ can be added to Nom(nominal) (ideally at the tail), recursively but complement make nominal into NP. 
# Adjunct will be added ideally at the tail of nominal but I regarded that adjective and nominal can also be just nominal. 
# GP, RelP, PastPartP, ToinfP is phrase starting with gerund, relational pronoun, past participle, to-infinitives , respectively.
# And rest are heuristically-added rules from train set like gerund phrase preceding sentence with comma…etc. 
# I didn’t use featured grammar because I thought it is not much effective in this case. 
# ‘SUBCAT’ feature, we don’t know the subcategory from pos tagger. TENSE or AGR or NUM feature didn’t seem to help a lot. 
# So I decided just to stick with the simple CFG.

# 4. Relation Extraction
# Now I have parsed tree with CFG, and possible five actions. For each action, find VP containing it. 
# If exist, we need to perform two tasks 1) find subject noun 2) find object noun. 
# For 2), find NP inside VP and find the very first noun traversed in that NP(because we only need to find one-word). 
# For 1), there are two cases – (1) VP is inside RelP(Phrase of which header is relative pronoun so modifying noun) 
# (2) VP is inside S(common case, S-> NP VP). 
# First find the nearest parent that is either RelP or S. 
# If RelP(case (1)) find NP that the RelP modifies and find noun from NP. 
# Else if S, find NP in that S and find noun from NP. Now we retrieved <noun,action,noun> triple.
#  But there can be too many triples. In this case, I got a heuristic rule. 
# ‘Choose the triple with shortest noun’.
#  It is because, the case when different triples resulted is mostly caused by the coordinate rule. 
# But I suggest it is inevitable. For the cases where, say, ‘and’ is used, it is ‘possible’ to read in many ways. 
# Although, what I observed is that if ‘clear’ and ‘make-sense’ phrases that is connected with conjunction exist, 
# the shortest noun most likely be that phrase(containing conjunction), not single noun without conjunction. 
# This improved the result quality drastically. 

#############



## tokenize raw sentence and pos tag
## additional process to match with the gramar, deal with poor POS tagger,...etc.
def toktag(raw):
    tags = []
    pairs = nltk.pos_tag(nltk.word_tokenize(raw))
    for idx, (word, tag) in enumerate(pairs):
        if word == "Do": tag = 'DO_Aux' 
        if tag.startswith('NN') or tag=='PRP': tag='NN'
        if tag.startswith('JJ') or tag=='PRP$': tag='JJ'
        if tag.startswith('RB'): tag='RB'
        if tag in ['VBP', 'VBZ', 'VBD', 'VB']: tag = 'VB'
        if tag=='WRB': tag = 'RB'
        if tag.startswith('W'): tag='WH' 
        if tag=='CC': tag='conj' # if word in ['or', 'and']: tag = 'conj'
       # if word=='that' and tag == 'IN': tag = 'ThatIN'
        if word.lower()=='that' : tag = 'That'
        ## too much problem with tagging 'that' --> IN / WP  is almost every usage -> aggregate to 'That'
        if word in ['be','was','were','is','am','are']: tag='beVB'
        if word =='whereas' : tag='conj'
        if tag =='FW': tag='NN'
        
        ##nltk pos tagger : often mistag Named Entity with Capitals as JJ
        ## but not the case when adjective is at the beginning of sentence
        if len(word)>2:
            if (not word[1:].islower()) and tag=='JJ' and (not word[:-2] == 'ed'): tag='NN'
        
        ###
        if word=='signal': tag = 'NN'
        if word=='reduces': tag ='VB'
        if word=='inhibits': tag = 'VB'
        if word=='inhibit': tag ='VB'
        if word=='point' : tag='VB'
        if word=='prevents' : tag = 'VB'
        if word=='binds': tag='VB'
        if word=='protein': tag = 'NN'
        if word== 'induces': tag='VB'
        if word== 'induced' and tag!='VBN': tag='VB'
        ###
        tags.append(tag)
     
        
    pairs = list(zip([word for word,tag in pairs] , tags))
        
    
    ## remove RB
    for i in range(len(pairs)-1):
        w1, t1 = pairs[i]
        w2, t2 = pairs[i+1]
        if t1=='RB' and w2==',':
            pairs[i]=('','')
            if i==0:
                pairs[i+1]=('','')
        elif w1==',' and t2=='RB':
            if (i+1==len(pairs)-1):
                pairs[i]=('','')
            pairs[i+1]=('','')
    pairs = [pair for pair in pairs if not pair==('','')]
    
    for i in range(len(pairs)):
        w1, t1 = pairs[i]
        if t1=='RB':
            pairs[i]=('','')
    pairs = [pair for pair in pairs if not pair==('','')]
    
    
    
    
    ## be p.p. by -> Make it into one token and tag it as 'Beppby'
    for i in range(len(pairs)-2):
        w1, t1 = pairs[i]
        w2, t2 = pairs[i+1]
        w3, t3 = pairs[i+2]
        if w1 in ['be','was','were','is','am','are'] and t2=='VBN' and w3=='by':
            pairs[i] = (w1+'-'+w2+'-'+w3, 'Beppby')
            pairs[i+1] = ('','')
            pairs[i+2] = ('','')
    pairs = [pair for pair in pairs if not pair==('','')]
    
    
    ## Dealing with poor POS tagger
    for idx, (word, tag) in enumerate(pairs):    
        if word=='prevented': tag='VB'
        if word=='inhibited': tag='VB'
    
    
    ##handle bywhich
    for i in range(len(pairs)-1):
        w1, t1 = pairs[i]
        w2, t2 = pairs[i+1]
        if w1=='by' and w2=='which':
            pairs[i] = (w1+' '+w2, 'Bywhich')
            pairs[i+1] = ('','')
    pairs = [pair for pair in pairs if not pair==('','')]
    
    
    ##remove parentheses
    for i in range(len(pairs)):
        if pairs[i][0] == '(':
            for j in range(i, len(pairs)):
                if pairs[j][0] == ')': break
            pairs[i:j+1] = [('','')]*(j+1-i)
    pairs = [pair for pair in pairs if not pair==('','')]
    
    
    ##phrasal verb - bind to
    for i in range(len(pairs)-1):
        w1, t1 = pairs[i]
        w2, t2 = pairs[i+1]
        if w1 in['bind','binds'] and w2=='to' and t1.startswith('VB'): 
            pairs[i] = (w1+' '+w2, 'VB')
            pairs[i+1] = ('','')
        if w1=='bound' and t1=='VBD' and w2=='to':
            pairs[i] = (w1+' '+w2, 'VB')
            pairs[i+1] = ('','')
            
    pairs = [pair for pair in pairs if not pair==('','')]
    
    return [pair for pair in pairs if not (pair[1]==':' or pair[1]=="\'\'") ]



grammar = nltk.CFG.fromstring("""
  CompleteS -> S '.' 

  S -> S 'conj' S | S ',' S | S ',' 'conj' S 
  NP -> NP 'conj' NP | NP ',' NP | NP ',' 'conj' NP  
  VP -> VP 'conj' VP | VP ',' VP | VP ',' 'conj' VP 
  AP -> AP 'conj' AP | AP ',' AP | AP ',' 'conj' AP 
  PP -> PP 'conj' PP | PP ',' PP | PP ',' 'conj' PP

                          
  S -> NP VP | Aux NP VP | 'WH' NP VP  | PP ',' S  |  S ',' PP | 'IN' S ',' S | AP ',' S | S ',' RelP | PastPartP ',' S | S ',' PastPartP | GP ',' S | S ',' GP
  Sbar -> 'That' S
  Aux -> 'MD' | 'DO_Aux'
  NP -> 'DT' Nom | Nom | GP | ToinfP | NP ',' NP ','
  Nom -> Nom PP | Nom RelP | Nom Sbar | Nom ',' RelP | Nom GP | Nom PastPartP | Nom 'CD' | 'CD' Nom | Nom ToinfP | Nom 'Bywhich' S |  AP Nom | Nom AP | Nom 'NN' | 'NN' 
  VP ->  Aux VP | 'Beppby' NP | Vom NP | Vom PP | Vom PastPartP | Vom S | Vom Sbar | Vom
  Vom -> 'VB' 'conj' 'VB' | 'VB'
  AP -> 'JJ'
  VP -> 'beVB' NP | 'beVB' PP | 'beVB' PastPartP | 'beVB' S | 'beVB' Sbar | 'beVB' AP | 'beVB'
  PP -> 'IN' NP | 'TO' NP | 'IN' PastPartP
  RelP -> 'WH' VP | 'That' VP
  GP -> 'VBG' NP | 'VBG' PP | 'VBG' S | 'VBG' Sbar| 'VBG'
  PastPartP -> 'VBN' NP | 'VBN' PP | 'VBN' S | 'VBN' Sbar| 'VBN'
  ToinfP -> 'TO' 'VB' NP | 'TO' 'VB' PP | 'TO' 'VB' S | 'TO' 'VB' Sbar | 'TO' 'VB'
  

""")
parser = nltk.ChartParser(grammar)
#parser = nltk.ChartParser(grammar_headonly, trace=2)



# Parse the raw sentence into parsed tree using CFG
def parse(raw):
    
    words = [word for word, tag in toktag(raw)]
    tags = [tag for word, tag in toktag(raw)]
    
    ## 1. parse by tag
    trees =[]
    for enum,tree in enumerate(parser.parse(tags)):
        trees.append(tree)
        
    ## 2. link to words using preorder traversal 
    new_trees=[]
    for tree in trees:
        tree_str = str(tree)
        words_copy = [word for word in words if not (word=='.' or word==',')]


        new_tree_str = add_words(tree_str, words_copy)
    
        new_tree= nltk.Tree.fromstring(new_tree_str)
        new_trees.append(new_tree)  
        
        
    return [ParentedTree.convert(new_tree) for new_tree in new_trees]

    

## adding corresp. word(lexicons) to pos_tag
def add_words(tree_str, words):
    terminals =  ['CompleteS', 'S', 'Ss', 'Sbar', 'NP', 'NPs', 'VP', 'VPs', 'Aux',  'AP', 'APs', 'Nom', 'VP',  'PP', 'RelP', 'GP', 'PastPartP','ToinfP', 'PPS','Vom']
    tag_list = tree_str.split(' ')
    tag_list = [tag.strip('\n') for tag in tag_list if tag!='']
    new_tag_list = []
    for tag in tag_list:
   
        real_tag=tag.strip('(').strip(')')
        if real_tag not in terminals+['.', ','] :
            word = words.pop(0)
            for i in range(len(tag)):
                if tag[i].isalpha():
                    break
            
            for j in range(len(tag)+1):
                if j==len(tag): break
                if tag[j]==')':
                    break
                
            prev=tag[:i]
            post=tag[j:]
            
            new_tag = prev+real_tag+'/'+word+post
            
            new_tag_list.append(new_tag)
        else:
            new_tag_list.append(tag)
            
    
    return ' '.join(new_tag_list)
    



##lemmatizer
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





##############################################################################################################
########### These are for internal traversals in parsed tree to find NE1, action, NE2 triples ################


## Search for action in the parsed tree
def search_action(tree, action):
    try:
       label= tree.label()
    except:
        return
        
    if type(tree[0]) == str and label=='Vom' :
        pos,word = tree[0].split('/')
        
        if pos=='VB' and lemmatize(word, 'V') == action:
            return tree
        
        
        if len(tree)==3 and tree[1].startswith('conj') and lemmatize(tree[2].split('/')[1],'V')==action:
            return tree
        
        
        
    else:
        if label=='VP' and type(tree[0]) == str:
            pos,word = tree[0].split('/')
            if pos=='Beppby':
                if lemmatize(word.split('-')[1] , 'V') == action:
                    return tree
        
        for child in tree:
            result = search_action(child, action)

            if result:
                return result


## find VP that contains action node
def find_VP(action_node):
    parent = action_node.parent()
    if parent.label() == 'VP':
        return parent
    else:
        return find_VP(parent)

## find NP that is inside VP
def find_NP(VP):
    for child in VP:
        try:
            label = child.label()
        except:
            continue
        if label == 'NP':
            return child
    
 #   print("findNP(VP)error")


## find S or RelP that contains VP
def find_S_or_Rel(VP):
    parent = VP.parent()
    if parent.label() == 'S':
        return (parent, True)
    if parent.label() == 'Rel':
        return (parent, False)
    return find_S_or_Rel(parent)


## find NP that is modified by RelP
def find_NP_from_RelP(RelP):
    parent=RelP.parent()
    if parent.label() == 'NP':
        return parent
    else: return find_NP_from_RelP(parent)



## find noun that is inside NP
def find_noun(NP):
    ## 1. conj 존재하면 나누고, 각각의 NP에 대해 find_noun()실행
    try:
        len(NP)
    except:
        return None
    if len(NP)>1 and (NP[1] == 'conj/and' or NP[1]=='conj/or'): #or NP[1]==','):
        try:
            result = find_noun(NP[0])+' '+ NP[1].split('/')[1] + ' '+find_noun(NP[2])
            return result
        except: 
            None
        ## 2. Nom에서 noun 하나만
    for child in NP:
        ## child에 NPs/NP 존재가능
        try:
            label=child.label()
            if (label=='Nom'):
                return find_NN(child)
            if (label=='NP'):
                return find_noun(child)
            if (label=='NPs'):
                for grandchild in child:
                    label=child.label()
                    if label=='NP':
                        return find_noun(grandchild)
                    elif label=='NPs':
                        return find_noun(grandchild)
            
            if (label=='GP'):
                return find_noun(child)
        except:
            None
        
## find noun that is inside Nom
def find_NN(Nom):
    for child in Nom:
        if type(child)==str:
            if child.startswith('NN'):
                return child[3:]
            else:continue
        else:
            if child.label()=='Nom':
                return find_NN(child)
  
        

## find triple from parsed tree
def get_triple_from_tree(tree,action):
    ##find action
    action_node = search_action(tree,action) # (Vom VB/activated) // VB 여러개 가능

    if action_node==None : return None

    if (len(action_node)==3): ## verb and verb
        action_annotate_sub1 = action_node[0].split('/')[1]
        action_annotate_sub2 = action_node[2].split('/')[1]
        if (lemmatize(action_annotate_sub1, 'V') == action):
            action_annotate = action_annotate_sub1
        elif (lemmatize(action_annotate_sub2,'V') == action):
            action_annotate = action_annotate_sub2
        else:
            return None

    elif len(action_node)==2 and action_node.label() =='Vom': ## binds to
        action_annotate = action_node[0].split('/')[1] + ' '+ action_node[1]
    else:
        action_annotate = action_node[0].split('/')[1]
    
    
    ##find noun_2
    if action_node[0].split('/')[0] == 'VB':
        VP = find_VP(action_node)  
    elif action_node[0].split('/')[0] == 'Beppby':
        VP = action_node
    else:
        None
      #  print('error')
    
    NP = find_NP(VP)
    noun2_annotate = find_noun(NP)
    
    ## find noun_1
    S_or_Rel, isS = find_S_or_Rel(VP)
    
    if isS:
        new_NP = find_NP(S_or_Rel)
        noun1_annotate = find_noun(new_NP)
        
    else:
        new_NP = find_NP_from_RelP(S_or_Rel)
        noun1_annotate = find_noun(new_NP)
        
    #print('result:',(noun1_annotate, action_annotate, noun2_annotate))
        
    action_annotate = ' '.join(action_annotate.split('-'))
        
    return (noun1_annotate, action_annotate, noun2_annotate)

    
##############################################################################################################    
##############################################################################################################
    
        
## Extract triples from raw sentence
def get_triple(raw):
    all_action_list = []
    
    parsed_trees= parse(raw)
 
    for action in ['activate', 'inhibit', 'bind', 'induce', 'prevent']:
        triples = set()
        triples = triples.union({get_triple_from_tree( tree,action) for tree in parsed_trees})
    
        if None in triples:
            triples.remove(None)
            
        triples = list( triple for triple in triples if not None in triple)
        
        ## remove pronouns
        new_triples = []
        for triple in triples:
            n1,act,n2 = triple
            if not (nltk.pos_tag([n1])[0][1].startswith('PRP') or nltk.pos_tag([n2])[0][1].startswith('PRP')):
                new_triples.append(triple)
            
            
        ## return the shortest
        ## ambiguity cased by conjunction is ineveitable -> but if shorter one exist, more likely be answer
                
        retList = sorted(new_triples, key=lambda x:(len(x[0]),len(x[2])) )
        if retList:
            all_action_list.append(retList[0])
        else:
            None
    if all_action_list:
        return all_action_list
    else:
        return [('','','')]
        


## relation extraction from sentences
def relation_extraction(raw_sentences):
    triples = []
    for idx,raw_sentence in enumerate(raw_sentences):
        print(idx)
        print(raw_sentence)
        triples.append(get_triple(raw_sentence))
    return triples


## read the annotated sentences from xlsx file. ## NOT USED
def read_from_xlsx():
    
    annotate_raws =[] # list of [tag, sent, pmid, year, corresp, publisher, triples]
    
    df = pd.read_excel (r'./CS372_HW4_sentences_20180368.xlsx')
    for i in range(df.shape[0]):
        annotate_raw=[]
        row = df.loc[i]
        if i%5==1:
            annotate_raw.append('test')
        else:
            annotate_raw.append('train')
        annotate_raw.append(row[0].strip('\xa0'))
        annotate_raw.append(row[1])
        annotate_raw.append(row[2])
        annotate_raw.append(row[3].strip('\xa0'))
        annotate_raw.append(row[4].strip('\xa0'))
        
        triple_1 = (row[5].strip('\xa0'),row[6].strip('\xa0'),row[7].strip('\xa0'))      
        
        if type(row[8])==str:
            triple_2 = ( row[8].strip('\xa0'), row[9].strip('\xa0'), row[10].strip('\xa0'))
        else:
            triple_2 = ('','','')
            
        annotate_raw.append([triple_1,triple_2])
        
        annotate_raws.append(annotate_raw)
    
    return annotate_raws

## read the annotated sentences from csv file.
def read_from_csv():
    
    annotate_raws=[]
    
    with open('CS372_HW4_sentences_20180368.csv', encoding='utf-8-sig') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for idx,row in enumerate(readCSV):
            if idx==0: continue
            annotate_raw=[]
            if idx%5==2: 
                annotate_raw.append('test')
            else: 
                annotate_raw.append('train')
            annotate_raw.append(row[0].strip('\xa0'))
            annotate_raw.append(row[1].strip('\xa0'))
            annotate_raw.append(row[2].strip('\xa0'))
            annotate_raw.append(row[3].strip('\xa0'))
            annotate_raw.append(row[4].strip('\xa0'))
            
            triple_1 = (row[5].strip('\xa0'),row[6].strip('\xa0'),row[7].strip('\xa0'))      
            
            if type(row[8])==str:
                triple_2 = ( row[8].strip('\xa0'), row[9].strip('\xa0'), row[10].strip('\xa0'))
            else:
                triple_2 = ('','','')
                
            annotate_raw.append([triple_1,triple_2])
            
            annotate_raws.append(annotate_raw)
    
    return annotate_raws
            
        

## relaxed evaluation between retrieved triples and correct triples
def relaxed_evaluation(rts,cts):
    
    num_retrieved = len([triple for rt in rts for triple in rt if not triple==('','','')])
    num_retrieved_correct = 0;
    num_correct = len([triple for ct in cts for triple in ct])
    
    for rt, ct in zip(rts,cts):
        if not rt==[('','','')]:
            for triple in rt:
                for triple_correct in ct:
                    if triple[0] in triple_correct[0] and triple[1]==triple_correct[1] and triple[2] in triple_correct[2]:
                        num_retrieved_correct+=1
                    else:
                        isFront = False
                        if 'and' in triple[0] and 'and' in triple_correct[0]:
                            lst = triple[0].split('and')
                            lst_c= triple_correct[0].split('and')
                            if lst[0] in lst_c[0] and lst[1] in lst_c[1]:
                                isFront = True
                        elif 'or' in triple[0] and 'or' in triple_correct[0]:
                            lst = triple[0].split('or')
                            lst_c= triple_correct[0].split('or')
                            if lst[0] in lst_c[0] and lst[1] in lst_c[1]:
                                isFront = True
                        elif triple[0] in triple_correct[0]:
                            isFront=True
                          
                        isBack = False
                        if 'and' in triple[2] and 'and' in triple_correct[2]:
                            lst = triple[2].split('and')
                            lst_c= triple_correct[2].split('and')
                            if lst[0] in lst_c[0] and lst[1] in lst_c[1]:
                                isBack = True
                        elif 'or' in triple[2] and 'or' in triple_correct[2]:
                            lst = triple[2].split('or')
                            lst_c= triple_correct[2].split('or')
                            if lst[0] in lst_c[0] and lst[1] in lst_c[1]:
                                isBack = True
                        elif triple[2] in triple_correct[2]:
                            isBack=True
                            
                        if isFront and isBack: num_retrieved_correct+=1
                        
    
    precision = num_retrieved_correct / num_retrieved
    recall = num_retrieved_correct / num_correct
    f_score = 2 * (precision*recall) / (precision+recall)
    print("precision:", precision)
    print("recall:", recall)
    print("f-score:", f_score)
    return precision, recall, f_score                        
            
            

## evaluate the relation extractor
def main_func():
    
    
    ##execute relation extraction module
    
    raw_sentences=[]
    correct_triples=[]
    
    raw_sentences_test = []
    correct_triples_test=[]
    
    tags=[]
    pmids=[]
    years=[]
    corresps=[]
    publishers=[]
    
    annotated_raws = read_from_csv()
    for annotated_raw in annotated_raws:     
        tag, raw, pmid, year, corresp, publisher, triples = annotated_raw
        triples = [triple for triple in triples if triple!=('','','')]
        
        raw_sentences.append(raw)
        correct_triples.append(triples)
        
        tags.append(tag)
        pmids.append(pmid)
        years.append(year)
        corresps.append(corresp)
        publishers.append(publisher)
        
        if (tag=='test'):
            raw_sentences_test.append(raw)
            correct_triples_test.append(triples)
 
    
    retrieved_triples = relation_extraction(raw_sentences)
    retrieved_triples_test = relation_extraction(raw_sentences_test)
    
    ##relaxed-evaluate and print f-score
    
    precision, recall, f_score = relaxed_evaluation(retrieved_triples_test, correct_triples_test)

    
    ##write  (original content)+expected triples to excel file 
    writer = pd.ExcelWriter('CS372_HW4_output_20180368.xlsx', engine='xlsxwriter')
     
    df = pd.DataFrame({'Tag': tags,'Sentences': raw_sentences, 'Correct': correct_triples , 'Extracted': retrieved_triples,
                   'PMID': pmids, 'Years' : years, 'CorrespAuthors' : corresps, 'Publishers' : publishers} )
    
    df.to_excel(writer)
    writer.save()

    data_xls = pd.read_excel('CS372_HW4_output_20180368.xlsx', index_col=None)
    data_xls.to_csv('CS372_HW4_output_20180368.csv', encoding='utf-8-sig', index=False)
    
    
    return


main_func()
        
            
