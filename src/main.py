from model import *
from preprocess import *

df = pd.read_csv("../data/hobbit1.csv", delimiter = ','); #hopefully not cheating
X, Y, vocab, word_to_index = CBOW_preprocess_training_data(df)

nn_model = CBOW_nn()
nn_model.fit(X, Y)

while(True): 
    context_words = []
    while(True):
        word = ''
        while(word not in word_to_index): 
            word = input()
            if((word in word_to_index) or (word == "END")): 
                break
            print("not in vocab")

        if(word == "END"):
            break
        
        print(f'added word: {word}')
        context_words.append(word)

    aggregated_cv = np.zeros((len(vocab), 1))

    for word in context_words: 
        aggregated_cv[word_to_index[word]] += 1.0 

    A0, A1, prediction = forward_prop(aggregated_cv, W0, W1, b0, b1)
    print(vocab[np.argmax(prediction.flatten(), axis = 0)])