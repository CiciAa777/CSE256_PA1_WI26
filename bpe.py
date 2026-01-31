# bpe.py
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

class BPE:
    """
    Minimal BPE (Sennrich-style) for word-level BPE:
    - Train on a corpus of words with counts
    - Learns a list of merges: (symbol_a, symbol_b) -> merged_symbol
    - Encode a word by applying merges in learned order
    """

    def __init__(self, num_merges: int = 2000):
        self.num_merges = num_merges
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

    @staticmethod
    def _word_to_symbols(word: str) -> Tuple[str, ...]:
        # end-of-word marker helps BPE learn word-final units
        return tuple(list(word) + ["</w>"])

    @staticmethod
    def _get_pair_counts(vocab: Counter) -> Counter:
        pair_counts = Counter()
        for symbols, freq in vocab.items():
            # symbols is a tuple of tokens like ('m','o','v','i','e','</w>')
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                pair_counts[pair] += freq
        return pair_counts

    @staticmethod
    def _merge_pair_in_word(symbols: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        a, b = pair
        merged = a + b
        new_syms = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
                new_syms.append(merged)
                i += 2
            else:
                new_syms.append(symbols[i])
                i += 1
        return tuple(new_syms)

    def train(self, words: List[str]):
        """
        Train BPE on a list of word tokens (strings), using their frequencies.
        """
        # Build initial vocab as Counter over symbol-tuples
        word_counts = Counter(words)
        vocab = Counter()
        for w, c in word_counts.items():
            vocab[self._word_to_symbols(w)] += c

        for merge_i in range(self.num_merges):
            pair_counts = self._get_pair_counts(vocab)
            if not pair_counts:
                break

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                # no meaningful merges left
                break

            # Apply merge to all words in vocab
            new_vocab = Counter()
            for symbols, freq in vocab.items():
                new_symbols = self._merge_pair_in_word(symbols, best_pair)
                new_vocab[new_symbols] += freq
            vocab = new_vocab

            self.merges.append(best_pair)

        # Rank merges for fast encoding
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    def encode_word(self, word: str) -> List[str]:
        """
        Encode a single word into BPE subwords using learned merges.
        Greedy merge based on learned merge ranks.
        """
        symbols = list(self._word_to_symbols(word))

        # Apply merges in rank order: repeatedly merge the best-ranked pair that exists
        while True:
            pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols) - 1)]
            ranked = [(self.merge_ranks[p], p) for p in pairs if p in self.merge_ranks]
            if not ranked:
                break
            _, best_pair = min(ranked)  # smallest rank = earliest learned merge

            # merge best_pair once across the whole sequence
            a, b = best_pair
            merged = a + b
            new_syms = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i+1] == b:
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(symbols[i])
                    i += 1
            symbols = new_syms

        # drop </w> marker in output tokens
        if symbols and symbols[-1] == "</w>":
            symbols = symbols[:-1]
        return symbols

    def encode_sentence(self, words: List[str]) -> List[str]:
        """
        Sentence is already tokenized into word tokens.
        Return a flat list of subword tokens.
        """
        pieces = []
        for w in words:
            pieces.extend(self.encode_word(w))
        return pieces
