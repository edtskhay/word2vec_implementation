class Vocabulary: 

    def __init__(self):
        self._vocab_list = None
        self._word_to_idx = None
        self._unique_word_count = 0

    def generate_vocab(self, tokens : list[list[str]] ) -> dict[str, int]:
        """
        Generates and initializes vocabulary's attribute given the tokens of a corpus
        :param tokens: A 2D list of tokens, each entry corresponds to a sentence, contains a list of each individual word within that sentence
        """ 

        self._vocab_list = sorted(set(word for sentence in tokens for word in sentence)) #get a sorted set of unique words
        self._word_to_idx = {word: idx for idx, word in enumerate(self._vocab_list)}
        self._unique_word_count = len(self._vocab_list)

    def get_vocab_list(self) -> list[str] | None:
        return self._vocab_list

    def get_word_to_idx(self) -> dict[str, int] | None:
        return self._word_to_idx

    def get_unique_word_count(self) -> int | None:
        return self._unique_word_count
