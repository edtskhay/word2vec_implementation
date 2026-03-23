import numpy as np 

from corpus import * 
from vocabulary import *
from utils import one_hot_generator

class CBOWDataset():

    def __init__(self, vocab : Vocabulary, corpus : Corpus, window_size : int): 

        self._window_size = window_size
        self._vocab = vocab
        self._corpus = corpus

        self._X, self._Y= self.generate_dataset()

    def generate_dataset(self):
        
        """
        Generates a dataset that satisfies the format of the model, namely X, Y = (example_count, vocab_size), according to the objects passed vocab and corpus attributes.
        Function slides a window of size (window_size * 2 + 1) across each sentence in the token list. For each window the following happens: 
            - Context words are converted into an aggregated (summed and averaged) context vector, forming an input example X 
            - Target words (centered within the window) are encoded into a one hot vector, forming output example Y
        
        It is worth mentioning the sliding window accomodate the boundaries of a sentence, fewer neighbors may be available than the specified window_size.
        
        :return: 
            X : A numpy array containing aggregated context vectors (example_count, vocab_size)
            Y : A corresponding numpy array containing one hot encoded target words (example_count, vocab_size)
        """

        #tokenized_sen = matricize_text(df)
        X_train = []
        y_train = []

        word_to_index= self._vocab.get_word_to_idx()
        tokenized_sentences = self._corpus.get_tokens()

        V = len(word_to_index)

        for sentence_list in tokenized_sentences: #at this point, we assume our data is a 2d array, of tokenized sentences, so iterate through those sen
            for i, target in enumerate(sentence_list): #slide through whole sentece, each word is a target word, no loop backs here for now

                aggregated_cv = np.zeros(V)

                start = max(0, i - self._window_size) #the start is capped to the first 
                end = min(len(sentence_list), i + self._window_size + 1) #end is capped to last element

                context = [sentence_list[j] for j in range(start, end) if j != i] #comprehension list, iterate through all words in our sublist window, keep target word out.
                for word in context: 
                    aggregated_cv[word_to_index[word]] += 1.0 

                if len(context) > 0:
                    aggregated_cv /= len(context)

                y_train.append(one_hot_generator(target, V, word_to_index)) #and append to context list to data matrix, and corresponding one hot target word rep in target list.
                X_train.append(aggregated_cv)

        return np.array(X_train), np.array(y_train) #works for our toy example
    
    def get_X(self) -> np.ndarray: 
        return self._X
    
    def get_Y(self) -> np.ndarray: 
        return self._Y