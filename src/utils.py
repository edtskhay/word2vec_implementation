import numpy as np

def calculate_cat_cross_entropy(Y, Y_predict):

    """
        Calculates CCE Loss given predicted probability distribution and target one hot vectors
        :param Y: True labels (one-hot encoded), shape (batch_size, vocab_size)
        :param Y_predict: Predicted labels (with applied softmax prob. distrib.), shape (batch_size, vocab_size)

        :return:
            loss: An output corresponding to the calculated loss between [0, +inf]
    """ 
    return - np.mean(np.sum(Y * np.log(Y_predict + 1e-12), axis=1))

def softmax(X : np.ndarray) -> np.ndarray:
        
    """
    Calculates softmax probability distribution vector given logit inputs 
    :param X: Input matrix (in this case coressponding to logit matrix output)

    :return: 
        X_softmaxxed: A transformed version of input matrix, with softmax activation function applied to each row
    """ 

    X_max = np.max(X, axis=1, keepdims=True) #retrieve row wise max value.
    e_X = np.exp(X - X_max) #subtraction by x_max to prevent overflow, conveniently softmax(x) = softmax(x - a) where a is max value from X 
    X_softmaxxed = e_X / np.sum(e_X, axis = 1, keepdims = True) 
    return X_softmaxxed

def print_summary(Y, Y_predict : np.ndarray, iteration : int): 

    """
    Prints training summary statistics: loss and top-k accuracies.

    :param Y: One-hot encoded true labels (batchsize, vocab_size)
    :param Y_predict: Predicted probabilities from model (batchsize, vocab_size)
    :param iteration: Current training iteration number (not epoch based)
    """
     
    loss = calculate_cat_cross_entropy(Y, Y_predict)

    top_10_predict_val = np.argsort(Y_predict, axis = 1)[:, -10:]
    top_5_predict_val = top_10_predict_val[:, -5:]
    top_1_predict_val = top_10_predict_val[:, -1:]

    true_val_index = np.argmax(Y, axis = 1)

    correct_in_10 = [true_val_index[i] in top_10_predict_val[i] for i in range(Y.shape[0])]
    correct_in_5 = [true_val_index[i] in top_5_predict_val[i] for i in range(Y.shape[0])]
    correct_in_1 = [true_val_index[i] in top_1_predict_val[i] for i in range(Y.shape[0])]
    
    top_10_accuracy = np.mean(correct_in_10)
    top_5_accuracy = np.mean(correct_in_5)
    top_1_accuracy = np.mean(correct_in_1)

    print(f'Iteration: {iteration}')
    print(f'CCE Loss: {loss:.5f}')
    print(f'Target in Top 1 Accuracy: {top_1_accuracy:.5f}')
    print(f'Target in Top 5 Accuracy: {top_5_accuracy:.5f}')
    print(f'Target in Top 10 Accuracy: {top_10_accuracy:.5f}')
    print()


def one_hot_generator(word : str, vocab_size : int , word_to_index : dict[str, int]) -> np.ndarray: 
    """
    Generates a one-hot encoded vector according to provided vocabulary

    :param word: The string to be encoded
    :param vocab_size: The size of the vocabulary
    :param word_to_index: hashmap (word, index), that maps each string in vocab to a corresponding index.
    :return:
        one_hot: a one hot vector, with the value corresponding to the provided words index being set to 1
    """

    one_hot = np.zeros(vocab_size)

    idx = word_to_index[word]
    if idx is not None: 
        one_hot[idx] = 1
    else: 
        raise ValueError(f'The word {word} is not present in the provided vocabulary')
    
    return one_hot

def shuffle_two_arrays_in_unison(X : np.ndarray, Y : np.ndarray): 
        
    """
    Shuffles two ndarrays with an identical permutation.

    :param word: The string to be encoded
    :param vocab_size: The size of the vocabulary
    :param word_to_index: hashmap (word, index), that maps each string in vocab to a corresponding index.
    :return:
        one_hot: a one hot vector, with the value corresponding to the provided words index being set to 1
    """

    perm = np.random.permutation(len(X))
    return X[perm], Y[perm]