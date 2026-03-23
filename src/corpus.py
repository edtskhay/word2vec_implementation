import pathlib as path 
import numpy as np 
import string


class Corpus: 

    def __init__(self, file_path : str = None):
        self._raw_lines = None
        self._word_count = 0
        self._tokens= None
        self._load_file(file_path)

    def _load_file(self, file_path : str):
        with open(file_path, 'r', encoding = 'utf-8') as f:
            self._raw_lines = f.read().splitlines()
            print(f"Generating tokens from corpus...")
            self._tokens= self.sanitize(self._raw_lines)
            self._word_count = sum(len(sentence) for sentence in self._tokens)

    def sanitize(self, raw_lines : list[str]) -> list[list[str]]: #god method, but itll be fine for now
        trans_table = str.maketrans('', '', string.punctuation)
        tokens = [line.translate(trans_table).strip().lower().split() for line in raw_lines]
        return tokens #this will be required in order to prevent windows from crossing between sentences.
    
    def get_raw_lines(self) -> list[str]:
        return self._raw_txt_list

    def get_word_count(self) -> int:
        return self._word_count

    def get_tokens(self) -> list[list[str]]:
        return self._tokens
