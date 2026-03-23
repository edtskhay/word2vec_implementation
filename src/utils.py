import numpy as np

def calculate_cat_cross_entropy(Y, Y_predict):
    return - np.mean(np.sum(Y * np.log(Y_predict + 1e-12), axis=1))

def softmax(X : np.ndarray) -> np.ndarray:
        X_max = np.max(X, axis=1, keepdims=True)
        e_X = np.exp(X - X_max)
        return e_X / np.sum(e_X, axis = 1, keepdims = True)

def print_summary(Y, Y_predict : np.ndarray, iteration : int): 
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
    print(f'Categorical Cross Entropy Loss: {loss:.5f}')
    print(f'Target in Top 1 Accuracy: {top_1_accuracy:.5f}')
    print(f'Target in Top 5 Accuracy: {top_5_accuracy:.5f}')
    print(f'Target in Top 10 Accuracy: {top_10_accuracy:.5f}')
    print()

def one_hot_generator(word : str, vocab_size : int , word_to_index : dict[str, int]) -> np.ndarray: 
    one_hot = np.zeros(vocab_size)

    idx = word_to_index[word]
    if idx is not None: 
        one_hot[idx] = 1
    else: 
        raise ValueError(f'The word {word} is not present in the provided vocabulary')
    
    return one_hot

def shuffle_two_arrays_in_unison(X, Y): 
        perm = np.random.permutation(len(X))
        return X[perm], Y[perm]