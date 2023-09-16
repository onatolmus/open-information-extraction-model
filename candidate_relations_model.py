import spacy
import benepar
nlp = spacy.load('en_core_web_md')  # load spacy's medium trained pipeline
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

import networkx as nx
from collections import Counter
import pardata
import pandas as pd
from itertools import compress
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag, word_tokenize, RegexpParser
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
import pickle


"""
Constraints:
    - There exists a dependency chain btw e1 and e2 no longer than 4. 
    - Dependency chain should contain some words of the relation r (usually the main verb).
    - Entities do not consist solely of the pronouns.
    - r should contain at least one VP tag.
    - r and object should have at least one VP tag as a common ancestor.
"""

def dep_chain_length(e1,e2,doc):  # e1 and e2 should be roots of noun chunks   
    """
    This function finds the shortest dependency chain path from roots of entities (noun chunks) 
    Inputs e1 and e2 represent roots of noun chunks. doc is the nlp tokens object created by spacy module.
    It returns the shortest path length and the chain itself if the path exists.
    """
    edges = [] ; 
    e2_index = None; e1_index = None
    for token in doc:
        for child in token.children:
            edges.append((token.lower_,child.lower_))
        if token.lower_ == e1.lower():
            e1_index = token.i
        if type(e1_index) == int and token.lower_ == e2.lower() and token.i>e1_index:
                e2_index = token.i
    if e2_index == None:
        return None, None
    graph = nx.Graph(edges) # create graph from the edges
    
    sp_length = nx.shortest_path_length(graph, source=str(e1.lower()), target=str(e2.lower()))
    chain = nx.shortest_path(graph, source=str(e1.lower()), target=str(e2.lower()))
       
    return sp_length, chain

def extract_potential_relation(chain):
    """
    Returns the candidate relation obtained by the chain. Entities (noun chunks) are subtracted.
    It is assumed that this would be a good relation between adjacent entities.
    """
    pot_relation = chain[1:-1]
    return pot_relation
    
def check_verb_in_chain(chain, pos_dict):
    """
    Constraint function to check for a verb within the chain.
    If verb exists in the chain then, it returns boolean TRUE, otherwise FALSE.
    """
    for link in chain:
        if (pos_dict[link] == 'VERB') or (pos_dict[link] == 'AUX'):
            return True
    return False

def check_pronouns(noun_chunk,pos_dict):
    """
    Check if noun chunk consists solely of pronouns
    Returns boolean.
    """
    nouns = noun_chunk.split()
    for noun in nouns:
        if noun.lower() in pos_dict.keys():
            if pos_dict[noun.lower()] != 'PRON':
                return True        
    return False

        
def VP_tag_relation(sent,relation):
    """
    This function takes the sentence and the potential relation as the input and checks 
    whether relation contains at least one VP tag in the constituency based tree.
    Returns a boolean.
    """
    for child in sent._.children:
        if 'VP' in child._.labels:
            for rel_word in relation:
                if rel_word in child.text:
                    return True
    return False

def common_VP_tag(sent,relation,e2):
    """
    This function takes the sentence, the potential relation and the right hand side entity in the triple as the input
    and checks whether relation and e2 contains a commong VP (Verb Phrase) tag in the constituency based tree.
    Returns a boolean.
    """
    for child in sent._.children:
        if 'VP' in child._.labels:
            for rel_word in relation:
                if rel_word in child.text and e2 in child.text:
                    return True
    return False




"""
The following section involves functions to create features to be used in building the model
presence of part-of-speech tag sequences in the relation 
the number of tokens in the potential relation
the number of stop words in potential relation 
whether an object e is found to be a proper noun, 
the part-of-speech tag to the left of e 
the part-of-speech tag to the right of e.
"""

def get_number_of_tokens(relation):
    return len(relation) # returns the length of the relation which is a list
        
def pos_tag_left(e,doc):
    """
    This function returns POS tag of the token located at the nearest left of the entity.
    To be used for the first entity.
    """
    pos_tag = "" ; left_index = None
    for token in doc:
        if token.lower_ == e.lower():
            left_index = token.i-1
            break
    if token.i != 0 and left_index != None:
        for tok in doc: 
            if tok.i == left_index:
                pos_tag = tok.pos_
                break
    return pos_tag
        
def pos_tag_right(e,doc):
    """
    This function returns POS tag of the token located at the nearest right of the entity. 
    To be used for the second entity.
    """
    pos_tag = "" ; right_index = None
    for token in doc:
        if token.lower_ == e.lower():
            right_index = token.i+1
            break
    if right_index != None:
        for tok in doc: 
            if tok.i == right_index:
                pos_tag = tok.pos_
                break
    return pos_tag

def check_proper_noun(e, pos_dict): # e should be root of noun chunk
    """
    This function checks if the root of the entity (noun chunk) is a proper noun or not.
    Returns boolean. Inputs are the root of the entity and the POS dictionary
    """
    prop_noun = False
    if e.lower() in pos_dict.keys():
        if pos_dict[e.lower()] == "PROPN":
            prop_noun = True
    return prop_noun

def number_of_stop_words(relation): # input is the pot relation (tokenized potential relation)
    """
    This function returns number of stop words inside the potential relation.
    nltk package is used for this purpose.
    """
    stop_words = set(stopwords.words('english'))
    sw_in_text = [word for word in relation if word in stop_words]
    return len(sw_in_text)

def pos_seq(relation,pos_dict):
    """
    This function returns POS sequence inside the relation.
    It is believed that it will be useful when used as a feature in the model
    """
    pos_seq_list = [pos_dict[word] for word in relation]
    return ' '.join(pos_seq_list) # returns a list of POS in order



"""
Next, I import the training dataset. For this purpose, GMB dataset is used. This dataset originally is to be used for Named Entity Recognition.
However, the algorithm does not need labelled data. I need decent amount of text in order for self supervised learner to have enough data
on judging whether the candidate relationship is trustworthy or not.

https://developer.ibm.com/exchanges/data/all/groningen-meaning-bank/
"""

def get_train_data_text():
    # pardata is the package that has the GMB dataset. Import it below, and split it as below
    data = pardata.load_dataset('gmb')['gmb_subset_full']
    df = pd.DataFrame([x.split(' ') for x in data.split('\n')], 
                      columns=['term', 'postags', 'entitytags'])
    
    
    # below operations are done to limit the number of training sentences to some amount. 
    # 5,000 sentences should be enough to train a Naive Bayes model.
    # I made use of the proposed data exploration approaches shown on the ibm's website
    sentence_count = 0
    sentence_count_list = []
    for i in range(len(df['term'])):
        if df['term'].iloc[i] == '.':
            sentence_count_list.append(sentence_count)
            sentence_count = sentence_count+1
        else:
            sentence_count_list.append(sentence_count)
    
    df['sentence_id'] = sentence_count_list
    data_text = ' '.join(df[df['sentence_id']<5000]['term'].tolist())
    return data_text

# next, I use nlp function to tokenize the training data
data_text = get_train_data_text()
doc = nlp(data_text)


def find_nc_tuples(nouns): 
    """
    This function finds pairs of entities that follow each other (nouns). We assume that most of the relations between two entities
    will appear between the text (relation,chain) of these two entities. For example if 4 entities appear in a sentence, we examine 
    the relations between e1 and e2, e2 and e3, and e3 and e4.
    """
    for index,_ in enumerate(nouns):
        if index + 2 <= len(nouns):
            yield (nouns[index],nouns[index+1])

def train_model(doc):
    """
    This functions trains the model using the tokenized document and returns the model, features and the column transformer.
    """

    # create a POS dictionary with keys representing the token and and values representing pos
    pos_dict_train=dict()
    for token in doc:
        pos_dict_train.update({token.lower_ : token.pos_})
    
    # initialize lists for features
    triples = [] ; labels = [] 
    no_of_tokens = []; pos_left = []; pos_right = []
    propn_1 = [] ; propn_2 = [] ;no_of_sw =[] ; pos_sequence = []
    for sent in doc.sents: # iterate over sentences
        chunk_dict = dict([(chunk.text, chunk.root.text) for chunk in sent.noun_chunks]) # dictionary for noun chunk and its root since some functions require the root
        chunk_list = [chunk.text for chunk in sent.noun_chunks] # this it to find the generator for the next for loop
        for chunks in find_nc_tuples(chunk_list): # loop over each adjacent noun chunk pair within a sentence
            chain = None 
            length, chain = dep_chain_length(chunk_dict[chunks[0]], chunk_dict[chunks[1]], sent)
            if chain != None: 
                pot_rel = extract_potential_relation(chain) # find the potential relations
                triples.append((chunks[0],pot_rel,chunks[1])) # append the triple with (e1, potential relation,e2)
                
                # append feature lists
                no_of_tokens.append(get_number_of_tokens(pot_rel))
                pos_left.append(pos_tag_left(chunks[0].split()[0],sent))
                pos_right.append(pos_tag_right(chunks[1].split()[-1],sent))
                propn_1.append(check_proper_noun(chunk_dict[chunks[0]],pos_dict_train))
                propn_2.append(check_proper_noun(chunk_dict[chunks[1]],pos_dict_train))
                no_of_sw.append(number_of_stop_words(pot_rel))
                pos_sequence.append(pos_seq(pot_rel,pos_dict_train))
                # check constraints for self supervised learner, assign 1 if trustworthy, assign 0 otherwise
                if (length <= 4 and check_verb_in_chain(chain, pos_dict_train) and check_pronouns(chunks[0],pos_dict_train) and 
                    check_pronouns(chunks[1],pos_dict_train) and VP_tag_relation(sent,pot_rel) and common_VP_tag(sent,pot_rel,chunks[1])):
                    labels.append(1)
                else:
                    labels.append(0)


    # use sklearn package and prepare features array with labels
    le = preprocessing.LabelEncoder() 
    enc_labels = le.fit_transform(labels) # encode the labels into 1 and 0s
    
    # for features, I use the column transformer for encoding. To encode, I use the ordinal encoder from sklearn. This is suitable for categorical variables

    # create a raw dataframe of not encoded features
    cols = ['no_of_tokens','pos_left','pos_right','propn_1','propn_2','no_of_sw','pos_sequence']
    df_features_raw = pd.DataFrame(list(zip(no_of_tokens,pos_left,pos_right,propn_1,propn_2,no_of_sw,pos_sequence)), columns = cols)
    
    # encode it using the column transformer and ordinal encoder
    ct_ore= make_column_transformer((OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), df_features_raw.columns))
    df_features = ct_ore.fit_transform(df_features_raw)

    # use sklearn naive bayes and use Categorical Naive to train the model
    model = CategoricalNB()
    # model trained below with the training features and encoded labels
    model.fit(df_features,enc_labels)
    return model, df_features, ct_ore



def get_candidate_relationships(doc_corp,entities,ct_ore,df_features,model):
    """
    Inputs --> doc_corp: the corpus(test) doc which is tokenized version by spacy, 
                entities: list of entities within the doc (sentence)
                ct_ore: column_transformer ordinal encoder, needed for transformation of the encoding
                df_features: needed for handling of the unknown variables. modes will be applied
                model: used for predicting the trustworthiness of the relationship by using Naive Bayes
    This function returns triples based on the trained model. The triples will include (entity1, candidate relationship, entity2)
    """
    pos_dict_corp=dict() # dictionary for the test corpus
    for token in doc_corp:
        pos_dict_corp.update({token.lower_ : token.pos_})
    
    
    # initialize lists for features
    corp_triples = [] ; corp_no_of_tokens = []; corp_pos_left = []; corp_pos_right = []
    corp_propn_1 = [] ; corp_propn_2 = [] ;corp_no_of_sw =[] ; corp_pos_sequence = []
    all_ents = []
    for sent in doc_corp.sents: #iterate over sentences
        for ents in find_nc_tuples(entities): 
            chain = None ; 
            length, chain = dep_chain_length(ents[0].root.text, ents[1].root.text, sent)
            if chain != None: 
                pot_rel = extract_potential_relation(chain)
                corp_triples.append((ents[0].text,pot_rel,ents[1].text))
                
                #append feature lists
                corp_no_of_tokens.append(get_number_of_tokens(pot_rel))
                corp_pos_left.append(pos_tag_left(ents[0].text.split()[0],sent))
                corp_pos_right.append(pos_tag_right(ents[1].text.split()[-1],sent))
                corp_propn_1.append(check_proper_noun(ents[0].root.text,pos_dict_corp))
                corp_propn_2.append(check_proper_noun(ents[1].root.text,pos_dict_corp))
                corp_no_of_sw.append(number_of_stop_words(pot_rel))
                corp_pos_sequence.append(pos_seq(pot_rel,pos_dict_corp))

    
    cols = ['no_of_tokens','pos_left','pos_right','propn_1','propn_2','no_of_sw','pos_sequence']
    df_features_corp_raw = pd.DataFrame(list(zip(corp_no_of_tokens,corp_pos_left,corp_pos_right,corp_propn_1,
                                             corp_propn_2,corp_no_of_sw,corp_pos_sequence)), columns = cols)
    
    if df_features_corp_raw.empty:
        return 
    
    df_features_corp = ct_ore.transform(df_features_corp_raw) # use the same encoding as the train model, column transformer and ordinal encoder enables this

    
    # there are some missing or unencountered values in the test features that was not seen in the train data.
    # for these, I replace NaNs with mode (most repeated term in the train features) of the feature. Most of the time this happens
    # with pos sequence, since there are a lot of unique possibilities for that.
    modes_dict = dict([(col,Counter(df_features[:,col]).most_common(1)) for col in range(df_features.shape[1])])
    for col in range(df_features_corp.shape[1]):
        df_features_corp[np.isnan(df_features_corp[:,col]),col]= modes_dict[col][0][0]
    
    
    predicted =  [None] * len(df_features_corp)
    for i in range(len(df_features_corp)):
        prediction = model.predict(df_features_corp[[i]])
        if prediction == 1:
            predicted[i] = True
        else:
            predicted[i] = False

    tuple_of_relations = list(compress(corp_triples, predicted))
    
    return tuple_of_relations



# train the model
model, df_features, ct_ore = train_model(doc)

# save the objects
with open('objects.pkl', 'wb') as f: 
    pickle.dump([model, df_features, ct_ore], f)







