#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:16:06 2019

@author: onlyone
"""
'''
import necessary libraries

Step-0: Tokenization (not a step because can be used in any step)

Step-1: Cleaning / Noise Removal
        a. HTML tags removal
        b. Accented characters removal: é to e
        c. Expanding Contractions: "don't" to "do not"
        d. Special Characters removal: #%
        e. Stopwords removal: a, an, the, ...
        f. Lower casing
        g. Punctuation removal
        h. Frequent words removal
        i. Rare words removal
        j. Spelling correction

Step-2: Normalization
        a. Stemming
        b. Lemmatization

Saving
'''
# Step-0: Tokenization

import nltk
nltk.download()
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize 

#------------------------    
# sentence tokenization
sentences = nltk.sent_tokenize(paragraph)
print(len(sentences))
sentences
#------------------------
# word tokenization
words=[nltk.word_tokenize(sent) for sent in sentences]
words
#------------------------
# regex tokenization
# Define a regex pattern to find hashtags: pattern1
pattern1 = r"#\w+"
# Use the pattern on the first tweet in the tweets list
regexp_tokenize(tweets[0], pattern1)
#------------------------
# Tweet tokenization
# to tokenize all tweets into one list
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
all_tokens = [tknzr.tokenize(t) for t in tweets]
print(all_tokens)

#****************************************************************

# 1.a: HTML tags removal
from bs4 import BeautifulSoup

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text
#*****************************************************************
    
# 1.b: Accented characters removal
import unicodedata

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text
#******************************************************************

# 1.c: Expanding Contractions 
# sample CONTRACTION_MAP: DJ-github
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
#*****************************************************************
    
# 1.d: Special Characters removal
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

remove_special_characters("Well this was fun! What do you think? 123#@!", 
                          remove_digits=True)
#******************************************************************

# 1.e: Stopwords removal
from nltk.corpus import stopwords
default_stopwords = stopwords.words('english') # or any other list of your choice

'''# alternative for defining stopwords

user_defined_stop_words = ['st','rd','hong','kong'] 
i = nltk.corpus.stopwords.words('english')
j = list(string.punctuation) + user_defined_stop_words
stopwords = set(i).union(j)
'''
def remove_stopwords(text, stop_words=default_stopwords):
    tokens = [w for w in tokenize_text(text) if w not in stop_words]
    return ' '.join(tokens)
#*****************************************************************

# 2.a: Stemming
from nltk.stem import PorterStemmer

def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text
#******************************************************************

# 2.b. Lemmatization

# with SpaCy
import spacy
nlp = spacy.load('en', disable=['parser', 'ner'])

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

# with NLTK
from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()
newsentences = nltk.sent_tokenize(paragraph)

for i in range(len(newsentences)):
    words = nltk.word_tokenize(newsentences[i])
    words = [lemmatizer.lemmatize(word) for word in words]
    newsentences[i] = ' '.join(words)
#******************************************************************

# Bringing it all together — Building a Text Normalizer    
def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_lemmatization=True, special_char_removal=True, 
                     stopword_removal=True, remove_digits=True):
    
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # strip HTML
        if html_stripping:
            doc = strip_html_tags(doc)
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        # expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
        # lowercase the text    
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
        # remove special characters and\or digits    
        if special_char_removal:
            # insert spaces between special characters to isolate them    
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)  
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
            
        normalized_corpus.append(doc)
        
    return normalized_corpus

'''# alternative approach for entire function
def preprocess(x):
    x = re.sub('[^a-z\s]', '', x.lower())                  # get rid of noise
    x = [w for w in x.split() if w not in set(stopwords)]  # remove stopwords
    and so on...
    return ' '.join(x)                                     # join the list
    '''
#******************************************************************

# Saving
news_df.to_csv('news.csv', index=False, encoding='utf-8')

#----------------------   E   N   D   -----------------------------