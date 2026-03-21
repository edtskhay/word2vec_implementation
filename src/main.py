from model import *
import os
from preprocess import *


print("Script __file__:", __file__)
print("Script directory:", os.path.dirname(__file__))

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/hobbit1.csv"), delimiter = ','); #hopefully not cheating
print("Reading from CSV file")
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

    aggregated_cv /= np.sum(aggregated_cv)
    prediction = nn_model.predict(aggregated_cv)

    print(vocab[np.argmax(prediction.flatten(), axis = 0)])