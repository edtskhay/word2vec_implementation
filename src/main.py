import os
import argparse

from word2vec import *
from corpus import *
from vocabulary import *
from dataset import *

def main(args): 

    #load corpus, and generate tokens
    file_path = os.path.join(os.path.dirname(__file__), args.file)
    print(f'Loading file from path {file_path}...')
    corpus = Corpus(file_path) #generates, and sanitizes txt file into tokens
    
    #generate vocabulary based on whatever is in the corpus
    vocab = Vocabulary()
    print('Generating vocabulary based on tokens...')
    vocab.generate_vocab(corpus.get_tokens())

    #generate the required dataset, with windowed context words paired with target words
    dataset = CBOWDataset(vocab, corpus, args.window)

    cbow_model = Word2Vec(vocab, args.context_dim, args.epochs, args.alpha)

    model_path = os.path.join(args.output, 'cbow_model.npy')
    
    if os.path.exists(model_path):
        print('Loading existing model...')
        cbow_model.load_from_file(model_path)
    else:
        print('Training model...')
        cbow_model.fit_model(dataset.get_X(), dataset.get_Y(), args.strategy)
        os.makedirs(args.output, exist_ok=True)
        cbow_model.save_to_file(model_path)
        print(f'Model saved to {model_path}')
    
    # Interactive prediction loop
    print('Enter (lowercase) words to predict the next word (type "END" to finish):')
    while(True): 
        context_words = []
        word_to_idx = vocab.get_word_to_idx()
        while(True):
            word = ''
            while(word not in word_to_idx):
                word = input("> ").strip()
                if(word == 'END'):
                    break
                if word not in word_to_idx: 
                    print('Inputted word not in vocabulary')
                    continue
                
                context_words.append(word)

            if(word == 'END'):
                break
            if not context_words:
                print('No words entered, exiting.')
                break
        
        if not context_words:
            break
        
        aggregated_cv = np.zeros((1, len(word_to_idx)))

        for word in context_words: 
            aggregated_cv[0, word_to_idx[word]] += 1.0 

        aggregated_cv /= np.sum(aggregated_cv)
        prediction = cbow_model.predict(aggregated_cv)
        certainty = np.max(prediction.flatten(), axis = 0)
        predicted_word = vocab.get_vocab_list()[np.argmax(prediction.flatten(), axis = 0)]
        print(f'With context words: {*context_words,}') #oh my god what is this syntax
        print(f'Predicted word: {predicted_word}')
        print(f'Certainty = {certainty:.2f}%')


parser = argparse.ArgumentParser(description='CBOW Word2Vec Prediction CLI')

# Required positional argument for the corpus file
parser.add_argument('file', type=str,
                    help='Target (txt) file to act as corpus')

# Optional arguments with defaults
parser.add_argument('--output', type=str, default='./output',
                    help='Directory to save model and outputs')
parser.add_argument('--strategy', type=str, choices=['vanilla', 'mini_batch'], default='vanilla',
                    help="Training strategy: 'vanilla' for full-batch SGD, 'mini_batch' for mini-batch SGD")
parser.add_argument('--batchsize', type=int, default=30,
                    help='Batch size used for mini-batch gradient descent (defaults to 30)')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of training epochs (defaults to 1000)')
parser.add_argument('--context_dim', type=int, default=30,
                    help='Dimension of context vector (defaults to 30)')
parser.add_argument('--alpha', type=float, default=0.01,
                    help='Learning rate for model updates (defaults to 0.01)')
parser.add_argument('--window', type=int, default=3,
                    help='Number of context words on each side of target word (defaults to 3)')
args = parser.parse_args()

if __name__ == '__main__': 
    main(args)