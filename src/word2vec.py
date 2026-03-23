import numpy as np
from utils import *

class NotFittedError(ValueError, AttributeError):
    pass

class Word2Vec:
    def __init__(self, vocab, embedded_dim : int = 100, epochs : int = 50, alpha = 0.1, batch_size = 50): 

        self._embedded_dim = embedded_dim
        self._alpha = alpha
        self._epochs = epochs
        self._vocab= vocab
        self._batch_size = batch_size

        self._W_hidden = np.random.randn(len(vocab.get_vocab_list()), embedded_dim)
        self._W_logit = np.random.randn(embedded_dim, len(vocab.get_vocab_list()))

    def save_to_file(self, file_path : str):
        with open(file_path, 'wb') as f:
            np.save(f, self._W_hidden)
            np.save(f, self._W_logit)

    def load_from_file(self, file_path : str):
        with open(file_path, 'rb') as f:
            self._W_hidden = np.load(f)
            self._W_logit = np.load(f)

    def forward_prop(self, X):
        #X : aggregated onehot input for group of words
        hidden = X @ self._W_hidden #U: intermediate output, context vectors (A1 : (samples_count, embedded_dim))
        logit = hidden @ self._W_logit #Prediction between vectors  (A2 : (sample_counts, vocab_size))
        Y_predict = softmax(logit) #Y: apply softmax to obtain final prediction, will be plugged into loss
        
        return hidden, logit, Y_predict
    
    def back_prop(self, X, Y, Y_predict, hidden):
        M = len(self._vocab.get_vocab_list())
        dLdu = Y_predict - Y # partial deriv dL/du  (shape is (M, vocab_size))
        dLdW_logit = (1 / M) * (hidden.T @ dLdu) #has to be same shape as w1, so (embedded, vocab)
        dLdW_hidden = (1 / M) * X.T @ (dLdu @ self._W_logit.T) 

        return dLdW_hidden, dLdW_logit
        
    def fit_model(self, X_train : np.ndarray, y_train : np.ndarray, strategy : str):
        vocab_len = X_train.shape[1]

        match strategy:
            case 'vanilla':
                return self.vanilla_GD(X_train, y_train)
            case 'mini_batch':
                return self.minibatch_GD(X_train, y_train)

    
    def predict(self, X_input): 
        predicted = self.forward_prop(X_input)[2]
        return predicted
    
    def update(self, dLdW_hidden, dLdW_logit):
        self._W_logit -= self._alpha * dLdW_logit
        self._W_hidden -= self._alpha * dLdW_hidden

    def vanilla_GD(self, X, Y):
        #verify that the model is fit TODO
        M, vocab_len = X.shape

        for i in range(self._epochs): 
            A1, A2, Y_predict = self.forward_prop(X)
            dLdW_hidden, dLdW_logit= self.back_prop(X, Y, Y_predict, A1)
        
            self.update(dLdW_hidden, dLdW_logit)

            if(i % 5 == 0):
                print_summary(Y, Y_predict, i)


    def minibatch_GD(self, X, Y): 
        print("Using mini_batch")
        M = X.shape[0]
        batch_count = M // self._batch_size 

        for i in range(self._epochs): 

            X_shuffled, Y_shuffled = shuffle_two_arrays_in_unison(X, Y)

            X_batch_list = [X_shuffled[j : j + self._batch_size] for j in range(0, M, self._batch_size)]
            Y_batch_list = [Y_shuffled[j : j + self._batch_size] for j in range(0, M, self._batch_size)]

            for j, (X_batch, Y_batch) in enumerate(zip(X_batch_list, Y_batch_list)):
                A1, A2, Y_predict = self.forward_prop(X_batch)
                dLdW_hidden, dLdW_logit= self.back_prop(X_batch, Y_batch, Y_predict, A1)
        
                self.update(dLdW_hidden, dLdW_logit)

                if (j % 5 == 0):
                    A1, A2, full_predict = self.forward_prop(X)
                    iteration_num = (i * len(X_batch_list)) + j
                    print_summary(Y, full_predict, iteration_num)

"""
class CBOW_nn_accelerated(CBOW_nn):
    def __init__(self, embedded_dim = 100, batch_size = 10, epochs = 50, alpha = 0.1):
        super().__init__(embedded_dim, epochs, alpha)
        self.batch_size = batch_size

    @staticmethod
    def shuffle_two_arrays_in_unison(X, Y): 
        perm = np.random.permutation(len(X))
        return X[perm], Y[perm]
        
        #combined = np.c_[X.reshape(len(X), -1), Y.reshape(len(Y), -1)]
        #np.random.shuffle(combined)
        #X_shuffled = combined[:, :X.size // len(X)].reshape(X.shape)
        #Y_shuffled = combined[:, X.size // len(X):].reshape(Y.shape)

    def fit(self, X_train, y_train):
        vocab_len = X_train.shape[1]
        self.init_vals(vocab_len)
        self.mini_batch_grad_desc(X_train, y_train)

    def mini_batch_grad_desc(self, X, Y): 
        self.check_is_fitted()
        M = X.shape[0]
        
        batch_count = M // self.batch_size 

        for i in range(self.epochs): 

            X_shuffled, Y_shuffled = CBOW_nn_accelerated.shuffle_two_arrays_in_unison(X, Y)

            X_batch_list = [X_shuffled[j : j + self.batch_size] for j in range(0, M, self.batch_size)]
            Y_batch_list = [Y_shuffled[j : j + self.batch_size] for j in range(0, M, self.batch_size)]

            for j, (X_batch, Y_batch) in enumerate(zip(X_batch_list, Y_batch_list)):
                A1, A2, Y_predict = self.forward_prop(X_batch, self.W0, self.W1, self.b0, self.b1)
                dLdW0, dLdW1, dLdb0, dLdb1 = self.back_prop(X_batch, Y_batch, Y_predict, A1, self.W1, self.batch_size)
        
                self.update_model(dLdW0, dLdW1, dLdb0, dLdb1)

                if (j % 5 == 0):
                    A1, A2, full_predict = self.forward_prop(X, self.W0, self.W1, self.b0, self.b1)
                    iteration_num = (i * len(X_batch_list)) + j
                    CBOW_nn.print_summary(Y, full_predict, iteration_num)
"""