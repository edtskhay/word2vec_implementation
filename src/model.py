import numpy as np

class NotFittedError(ValueError, AttributeError):
    pass

class CBOW_nn: 
    def __init__(self, embedded_dim = 100, epochs = 50, alpha = 0.1): 
        self.embedded_dim = embedded_dim
        self.alpha = alpha
        self.epochs = epochs

        self.W0 = None
        self.W1 = None 
        self.b0 = None #in principle never used, no point offsetting contexts?
        self.b1 = None 

    def fit(self, X_train, y_train):
        vocab_len = X_train.shape[1]
        self.init_vals(vocab_len)
        self.vanilla_grad_desc(X_train, y_train)
    
    def predict(self, X_input): 
        self.check_is_fitted()
        predicted = self.forward_prop(X_input, self.W0, self.W1, self.b0, self.b1)[2]
        return predicted
    
    def check_is_fitted(self):
        if not self.is_fitted():
            raise NotFittedError

    def is_fitted(self):
        return all(v is not None for v in [self.W0, self.W1, self.b0, self.b1])

    
    def init_vals(self, vocab_len): #decision to be made, pass vocab into fit, probs.
        self.W0 = np.random.randn(vocab_len, self.embedded_dim) #matrix to map to trait nodes
        self.W1 = np.random.randn(self.embedded_dim, vocab_len) #
        self.b0 = np.zeros((1, self.embedded_dim))
        self.b1 = np.zeros((1, vocab_len))

    def forward_prop(self, X, W0, W1, b0, b1):
        #X : aggregated onehot input for group of words
        A1 = X @ W0 + b0 #U: intermediate output, context vectors (A1 : (samples_count, embedded_dim))
        #no activation function on hidden layer output  
        A2 = A1 @ W1 + b1 #Prediction between vectors  (A2 : (sample_counts, vocab_size))
        Y_predict= CBOW_nn.softmax(A2) #Y: apply softmax to obtain final prediction, will be plugged into loss
        return A1, A2, Y_predict

    def back_prop(self, X, Y, Y_predict, A1, W1, M):
        dLdb1, dLdb0 = 0, 0
        dLdu = Y_predict - Y # partial deriv dL/du  (shape is (M, vocab_size))
        dLdW1 = (1 / M) * (A1.T @ dLdu) #has to be same shape as w1, so (embedded, vocab)
        dLdb1 = (1 / M) * (np.sum(dLdu, axis = 0, keepdims=True)) 
        dLdW0 = (1 / M) * X.T @ (dLdu @ W1.T) 

        return dLdW0, dLdW1, dLdb0, dLdb1
    
    
    def update_model(self, dLdW0, dLdW1, dLdb0, dLdb1):
        self.W1 -= self.alpha * dLdW1
        self.W0 -= self.alpha * dLdW0

        self.b1 -= self.alpha * dLdb1 
        self.b0 -= self.alpha * dLdb0
  
    
    def vanilla_grad_desc(self, X, Y):
        #verify that the model is fit TODO
        self.check_is_fitted()

        M, vocab_len = X.shape

        for i in range(self.epochs): 
            A1, A2, Y_predict = self.forward_prop(X, self.W0, self.W1, self.b0, self.b1)
            dLdW0, dLdW1, dLdb0, dLdb1 = self.back_prop(X, Y, Y_predict, A1, self.W1, M)
        
            self.update_model(dLdW0, dLdW1, dLdb0, dLdb1)

            if(i % 5 == 0):
                CBOW_nn.print_summary(Y, Y_predict, i)
    
    
    @staticmethod
    def softmax(X):
        X_max = np.max(X, axis=1, keepdims=True)
        e_X = np.exp(X - X_max)
        #something about numerical stability will add later if i understand it 
        return e_X / np.sum(e_X, axis = 1, keepdims = True) #dumb luck

    @staticmethod
    def print_summary(Y, Y_predict, iteration): 
        loss = -np.mean(np.sum(Y * np.log(Y_predict + 1e-12), axis=1))
    

        top_10_predict_val = np.argsort(Y_predict, axis = 1)[:, -10:]
        true_val_index = np.argmax(Y, axis = 1)
        
        correct_in_10 = [true_val_index[i] in top_10_predict_val[i] for i in range(Y.shape[0])]
        top_10_accuracy = np.mean(correct_in_10)

        print(f'iteration: {iteration}')
        print(f"loss: {loss}")
        print(f"top_10_accuracy: {top_10_accuracy}")

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
            
    