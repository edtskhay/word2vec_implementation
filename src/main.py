from model import *
import os
from preprocess import *


print("Script __file__:", __file__)
print("Script directory:", os.path.dirname(__file__))

print(os.path.join(os.path.dirname(__file__), "../data/hobbit1.csv"))
csv_path = os.path.join(os.path.dirname(__file__), "../data/hobbit1.csv")
df = pd.read_csv(csv_path, delimiter =",")

print("Reading from CSV file")
X, Y, vocab, word_to_index = CBOW_preprocess_training_data(df)

#nn_model = CBOW_nn_accelerated(epochs = 10000, batch_size = 50)
nn_model = CBOW_nn_accelerated(epochs = 1000, batch_size= 10, alpha = 0.01)
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

    aggregated_cv = np.zeros(1, len(vocab))

    for word in context_words: 
        aggregated_cv[word_to_index[word]] += 1.0 

    aggregated_cv /= np.sum(aggregated_cv)
    prediction = nn_model.predict(aggregated_cv)

    print(vocab[np.argmax(prediction.flatten(), axis = 1)])