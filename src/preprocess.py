import numpy as np
import pandas as pd

df_corpus = pd.read_csv("data/hobbit1.csv")
df_corpus.drop(columns=["name"], inplace=True)



