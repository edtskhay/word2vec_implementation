# Word2Vec CBOW Scratch Implementation

Developed for the 2026 Summer/Autumn Internship Application Round. 

An implementation of a Word2Vec CBOW architecture using nothing but Numpy and grit.


Includes a "proof of concept" CLI to allow the user to train/load and subsequently use the model to predict a "target word" given a list of "context words". 

## Implemented features
* **CBOW Architecture**: Provided with a set of context words, the model attempts to predict a central target word. 
* **Softmax Activation**: The model's current activation function performs standard Softmax across the provided example batch. 

* **Optimization Strategy**: Currently two configurable optimization strategies have been implemented ie. Vanilla and Mini-Batch Gradient Descent.

* **Loss and Backpropogation**: CCE Loss function is utilized to during backpropogation to iteratively update the model weights. 

* **Preprocessing**: Automatic generation of corpus, vocabulary and corresponding dataset given a .txt file, each line corresponding to a sentence.

* **Configurablity**: Almost all parameters and configurations can be manipulated using the corresponding flag.

* **Automatic model loading and export**: Model weights are exported to a specified location. Attempting to pass the same path on new run will automatically load the preexisitng waits to the model.

## Requirements

- Python 3.8+
- NumPy

Install dependencies:

```bash
pip install numpy
```

## Usage

Run the CLI on the terminal:
```python main.py <corpus_file> [options]```

Feel free to use ```../data/hobbit_script.txt``` as a toy corpus_file. Copyright JR Tolkien.

After the training has been completed, or a model as been loaded, type in the context words one by one. To indicate that you have finished adding context words, input **END**. The model will return a predicted target word with a certainty. 

To exit the program, type **END**, without having added any context words.


### Positional Arguments

| Argument       | Description                                  |
|----------------|----------------------------------------------|
| `corpus_file`  | Path to the target `.txt` file to be used as the corpus |


### Optional Arguments

| Flag            | Type  | Default         | Description                                                                                  |
|-----------------|-------|----------------|----------------------------------------------------------------------------------------------|
| `--output_dir`  | str   | `./output`     | Directory of the output file to which trained model will be saved to.                                                |
| `--output_name` | str   | `cbow_model`   | Name of the output file to which trained model will be saved to.                             |
| `--strategy`    | str   | `vanilla`      | Training strategy: `'vanilla'` for full-batch SGD, `'mini_batch'` for mini-batch SGD.        |
| `--batchsize`   | int   | 30             | Batch size used for mini-batch gradient descent.                                             |
| `--epochs`      | int   | 1000           | Number of training epochs.                                                                   |
| `--context_dim` | int   | 30             | Dimension of the context vector.                                                             |
| `--alpha`       | float | 0.01           | Learning rate for model updates.                                                             |
| `--window`      | int   | 3              | Number of context words on each side of the target word.                                     |

### Examples

**1. Basic usage with defaults:**

```bash
python main.py corpus.txt
```

**2. Specify output directory and file:**

```bash
python main.py corpus.txt --output_dir ./my_model --output_file
```

**3. Mini-batch training with batch size 50:**

```bash
python main.py corpus.txt --strategy mini_batch --batchsize 50
```

**4. Custom hyperparameters:**

```bash
python main.py corpus.txt \
    --epochs 500 \
    --alpha 0.05 \
    --window 5 \
    --context_dim 50
```

### Corpus Format

- The corpus should be a plain text file (`.txt`).  
- Each **sentence should ideally be on a separate line**.  
- Example:

```txt
My dear Frodo.
You asked me once if I had told you everything there was to know about my adventures.
I am old now, Frodo.
I’m not the same Hobbit I once was.
```
# Potential Improvements and Extensions

* Performing a softmax on the entire provided batch is a suboptimal and extremely expensive strategy. Considering d embedding dimensions, a vocabulary size of V, classic softmax evaluates to a runtime of O(d * V) per example. Hence some alternatives come to mind:

    - Negative sampling, stands to decrease our runtime to O(d * (k + 1)) per example, where k + 1 corresponds to the amount of negative samples used during training, including the target word. As k, relatively speaking, is signficantly smaller than V, this modficiation would significantly accelerate training. 

    - Hierarchcal softmax, proves as another suitable alternative. Organizing the vocabulary into a huffman tree, instead of evaluating the entire list, would reduce the runtime of training to O(d * log(V)) per example. 

* Currently, only two optimizers are available, both of which rely on a constant learning rate. Incorporating simple momentum-based strategies, such as Nesterov momentum or Adam, would likely improve convergence speed during training. 

* Evaluation could be further enhanced with the generation of simple visualization. Plots of loss curves, accuracy trends across the training time would not only serve to build a more complete project, but allow me to ultimately tune the many configurable hyperparameters I have introduced in a more streamlined manner.
