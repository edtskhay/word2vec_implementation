import numpy as np
import string 

import pandas as pd

sanitizer = lambda x: x.replace('‘', '').replace('…', '').replace(";", "").replace("—", " ").translate(str.maketrans('', '', string.punctuation)).strip().lower() #patchwork ass sanitization function that removes leading spaces, puncutation and replaces - with a space
#clean_script_list = np.loadtxt("../data/hobbit1.csv", delimiter=',', usecols = 1,  skiprows = 1, dtype=str, converters = sanitizer, encoding="utf-8")
#this sucks real bad, and doesnt detect greek punctuation so im throwing it out of the window and using a df.

df = pd.read_csv("../data/hobbit1.csv", delimiter = ','); #hopefully not cheating
clean_script_list = [sanitizer(line) for line in df["line"].tolist()]; 

tokenized_script = [sentence.split() for sentence in clean_script_list] #split into 2d array of sentences split into words

vocab = sorted(set(word for sentence in tokenized_script for word in sentence)) #get a sorted set of unique words
word_to_index = {word: idx for idx, word in enumerate(vocab)} #initialize a dict, word corresponds to a value index

def one_hot_generator(word, vocab_size): 
    one_hot = np.zeros(vocab_size)
    one_hot[word_to_index[word]] = 1 
    return one_hot

def matricize_text(script_df = None):
    clean_script_list = [sanitizer(line) for line in df["line"].tolist()]; 
    tokenized_script = [sentence.split() for sentence in clean_script_list]
    return tokenized_script

def generate_vocab(tokenized_script):
    return sorted(set(word for sentence in tokenized_script for word in sentence)) #get a sorted set of unique words

def generate_word_to_idx(vocab): 
    return {word: idx for idx, word in enumerate(vocab)}


def CBOW_preprocess_training_data(df, window_size = 3):

    tokenized_sen = matricize_text(df)
    vocab = generate_vocab(tokenized_script)
    word_to_index = generate_word_to_idx(vocab)

    X_train = []
    y_train = []

    V = len(word_to_index)

    for sen in tokenized_sen: #at this point, we assume our data is a 2d array, of tokenized sentences, so iterate through those sen
        for i, target in enumerate(sen): #slide through whole sentece, each word is a target word, no loop backs here for now

            aggregated_cv = np.zeros(V)

            
            start = max(0, i - window_size) #the start is capped to the first 
            end = min(len(sen), i + window_size) #end is capped to last element

            context = [sen[j] for j in range(start, end) if j != i] #comprehension list, iterate through all words in our sublist window, keep target word out.
            for word in context: 
                aggregated_cv[word_to_index[word]] += 1.0 

            if len(context) > 0:
                aggregated_cv /= len(context)

            y_train.append(one_hot_generator(target, V)) #and append to context list to data matrix, and corresponding one hot target word rep in target list.
            X_train.append(aggregated_cv)

    return np.array(X_train).T, np.array(y_train).T, vocab, word_to_index