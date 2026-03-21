import numpy as np

class NotFittedError(ValueError, AttributeError):
    pass

class CBOW_nn: 
    def __init__(self, embedded_dim = 100, batch_size = 10, epochs = 50, alpha = 0.1): 
        self.embedded_dim = embedded_dim
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs

        self.W0 = None
        self.W1 = None 
        self.b0 = None #in principle never used, no point offsetting contexts?
        self.b1 = None 

    def fit(self, X_train, y_train):
        vocab_len = X_train.shape[0]
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
        self.b0 = np.zeros((self.embedded_dim, 1))
        self.b1 = np.zeros((vocab_len, 1))

    def forward_prop(self, X, W0, W1, b0, b1):
        #X : aggregated onehot input for group of words
        A1 = W0.T @ X+ b0 #U: intermediate output, context vectors
        #no activation function on hidden layer output  
        A2 = W1.T @ A1 + b1 #Prediction between vectors 
        Y_predict= CBOW_nn.softmax(A2) #Y: apply softmax to obtain final prediction, will be plugged into loss
        return A1, A2, Y_predict

    def back_prop(self, X, Y, Y_predict, A1, W1, M):
        dLdb1, dLdb0 = 0, 0
        dLdu = Y_predict - Y # partial deriv dL/du
        dLdW1 = (1 / M) * (A1 @ dLdu.T)
        dLdb1 = (1 / M) * (dLdu @ np.ones((M, 1)))
        dLdW0 = (1 / M) * (X @ (W1 @ dLdu).T)

        return dLdW0, dLdW1, dLdb0, dLdb1
    
    
    def update_model(self, dLdW0, dLdW1, dLdb0, dLdb1):
        self.W1 -= self.alpha * dLdW1
        self.W0 -= self.alpha * dLdW0

        self.b1 -= self.alpha * dLdb1 
        self.b0 -= self.alpha * dLdb0
  
    
    def vanilla_grad_desc(self, X, Y):
        #verify that the model is fit TODO
        self.check_is_fitted()

        vocab_len, M = X.shape

        for i in range(self.epochs): 
            A1, A2, Y_predict = self.forward_prop(X, self.W0, self.W1, self.b0, self.b1)
            dLdW0, dLdW1, dLdb0, dLdb1 = self.back_prop(X, Y, Y_predict, A1, self.W1, M)
        
            self.update_model(dLdW0, dLdW1, dLdb0, dLdb1)

            if(i % 5 == 0):
                CBOW_nn.print_summary(Y, Y_predict, i)
    
    
    @staticmethod
    def softmax(X):
        X_max = np.max(X, axis=0, keepdims=True)
        e_X = np.exp(X - X_max)
        #something about numerical stability will add later if i understand it 
        return e_X / np.sum(e_X, axis = 0, keepdims = True) #dumb luck

    @staticmethod
    def print_summary(Y, Y_predict, iteration): 
        loss = -np.mean(np.sum(Y * np.log(Y_predict + 1e-12), axis=0))
        true_val = np.argmax(Y, axis = 0)

        top_10_predict_val = np.argsort(Y_predict, axis = 0)[-10:]
        true_val_index = np.argmax(Y, axis = 0)
        
        correct_in_10 = [true_val_index[i] in top_10_predict_val[:, i] for i in range(Y.shape[1])]
        top_10_accuracy = np.mean(correct_in_10)

        print(f'iteration: {iteration}')
        print(f"loss: {loss}")
        print(f"accuracy: {top_10_accuracy}")