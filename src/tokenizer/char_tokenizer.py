class CharTokenizer:
    def __init__(self, text):
        self.text = text
        self.vocab = sorted(list(set(self.text)))
        self.vocab_size = len(self.vocab)

        # Create character to integer and integer to character mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}

    def encode(self, text):
        """
        Convert text to a list of integers based on the vocabulary.

        Args:
            text (str): Input text to encode

        Returns:
            list: List of integers representing the encoded text
        """
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        """
        Convert a list of integers back to text.

        Args:
            indices (list): List of integers to decode

        Returns:
            str: Decoded text
        """
        return "".join([self.idx_to_char[idx] for idx in indices])

    def get_vocab_size(self):
        """
        Get the size of the vocabulary.

        Returns:
            int: Size of the vocabulary
        """
        return self.vocab_size

    def get_vocab(self):
        """
        Get the vocabulary as a list.

        Returns:
            list: List of characters in the vocabulary
        """
        return self.vocab
