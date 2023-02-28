'''
Created by Jingkai Wang
'''

import re
import nltk
import json
from nltk.corpus import stopwords
from collections import Counter


def json_parser(filename):
    f = open(filename, 'r')
    raw = json.load(f)
    text = raw['text']
    words = text.split(" ")
    wc = Counter(words)
    num = len(words)
    f.close()
    return {'wordcount': wc, 'numwords': num}

def remove_punctuation(text):
    '''
    remove the punctuation of the text file
    @param: text (str): the whole text
    @return: clean_text (str): the text without punctuation
    '''
    clean_text = re.sub(r'[^\w\s]', '', text).strip()
    return clean_text

def tokenize(text):
    '''
    split text into list of the text file
    @param: text (str): the whole text
    @returnL tokens (list): the list that contains every split words
    '''
    # split text into single lists
    tokens = nltk.word_tokenize(text)

    # make all the words in the text file separatly
    tokens = [token.strip() for token in tokens] 
    return tokens 

def load_stop_words():
    '''
    load the stop words with NLTK library for those most common word in english,
    for example: but, same, the, a, etc.
    @return: stop_words (list): the list of stop words
    '''
    # Read a list of common or stop words. These get filtered from each file automatically.
    stop_words = set(stopwords.words('english'))
    return stop_words

def remove_stop_words(tokens):
    '''
    remove the stop words from tokens words
    @param: tokens (list): the list of every single words
    @return: no_stopword_tokens (list): the list that has been removed the stop words
    '''
    stop_words = load_stop_words() # call the function above
    no_stopword_tokens = [] # create a new list

    # fill the words without stop words into the new list
    for w in tokens:
        if w not in stop_words:
            no_stopword_tokens.append(w)

    return no_stopword_tokens

def capitalize(tokens):
    ''' convert all letters to upper case '''
    tokens = [token.upper() for token in tokens]
    return tokens

def get_avg_word_length(tokens):
    '''
    calculate the average length of the words
    @param: tokens (list): the list of every single words
    @return: length_sum/len(tokens)
    '''
    length_sum = 0 # initial the length_sum
    for t in tokens:
        length_sum += len(t) # calculate the words length and then add together
    return length_sum/len(tokens)

def calc_sentiment_score(tokens, pos_file="positive-words.txt", neg_file="negative-words.txt"):
    '''
    calculate the sentiment score of the text words
    @param:
        tokens (list): the list of every single words
        pos_file(txt): positive-words.txt
        neg_file(txt): negative-words.txt
    @return:
        score(float): the sentiment score
    '''
    pos_words = []
    neg_words = []

    f_pos = open(pos_file, 'r') # open the positive words text file
    for line in f_pos.readlines():
        pos_words.append(line.strip().upper()) # fill in with uppercase words

    f_neg = open(neg_file, 'r') # open the negative words text file
    for line in f_neg.readlines():
        neg_words.append(line.strip().upper()) # fill in with uppercase words

    # calculate the score
    score = 0
    for word in tokens:
        if word in neg_words:
            score -= 1
        if word in pos_words:
            score += 1
    return score

def general_text_parser(filename):
    '''
        Pre-process the file including removing unnecessary whitespace, stop words, punctuation,
    and capitalization.Then store intermediate results: statistics such as word count,
    average word length and sentiment.

    @param:
        filename (str): the name of the file we need to use
    @return:
        a dcitionary: {word_count: X, word_length: Y, sentiment_score: Z}
    '''

    text = open(filename, 'r').read()
    clean_text = remove_punctuation(text)
    tokens = tokenize(clean_text)
    no_stopword_tokens = remove_stop_words(tokens)
    final_tokens = capitalize(no_stopword_tokens)

    wc = len(final_tokens)
    avg_length = get_avg_word_length(final_tokens)
    sent_score = calc_sentiment_score(final_tokens)

    return {'word_count': wc, 'word_length': avg_length, 'sentiment_score': sent_score}



