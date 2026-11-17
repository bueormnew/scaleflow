"""
Byte-Pair Encoding (BPE) Tokenizer for ScaleFlow.

A clean, standalone, zero-dependency BPE tokenizer implementation with:
- Subword vocabulary generation and merge rule learning.
- Special tokens: <pad>, <bos>, <eos>, <unk>.
- Fast cached tokenization and lossless text reconstruction.
- JSON serialization (save / load).
- PyTorch compatibility helper for batch encoding and attention masks.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
except ImportError:
    torch = None

# Regex pattern splitting text into subword chunks while preserving whitespace
# via the meta-character ' ' (U+2581 LOWER ONE EIGHTH BLOCK)
_CHUNK_PATTERN = re.compile(r" ?[^\s ]+| +|\s+")
_META_SPACE = " "


class BatchEncoding(dict):
    """
    Dictionary-like container for batch tokenization results, providing
    attribute-style access and PyTorch device placement helpers.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__()
        if data:
            self.update(data)
        self.update(kwargs)

    @property
    def input_ids(self) -> Any:
        return self.get("input_ids")

    @property
    def attention_mask(self) -> Any:
        return self.get("attention_mask")

    def __getattr__(self, item: str) -> Any:
        if item in self:
            return self[item]
        raise AttributeError(f"'BatchEncoding' object has no attribute '{item}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to(self, device: Any) -> "BatchEncoding":
        """
        Moves all torch.Tensor items in this batch to the target device.
        """
        new_data: Dict[str, Any] = {}
        for k, v in self.items():
            if torch is not None and isinstance(v, torch.Tensor):
                new_data[k] = v.to(device)
            else:
                new_data[k] = v
        return BatchEncoding(new_data)


class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer.

    Attributes:
        vocab (Dict[str, int]): Token to ID mapping.
        id_to_token (Dict[int, str]): ID to token mapping.
        merges (List[Tuple[str, str]]): Ordered list of learned BPE merge rules.
        bpe_ranks (Dict[Tuple[str, str], int]): Priority rank of each merge rule.
        pad_token (str): Special padding token.
        bos_token (str): Special beginning-of-sequence token.
        eos_token (str): Special end-of-sequence token.
        unk_token (str): Special unknown token.
    """

    DEFAULT_PAD = "<pad>"
    DEFAULT_BOS = "<bos>"
    DEFAULT_EOS = "<eos>"
    DEFAULT_UNK = "<unk>"

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[Sequence[Tuple[str, str]]] = None,
        pad_token: str = DEFAULT_PAD,
        bos_token: str = DEFAULT_BOS,
        eos_token: str = DEFAULT_EOS,
        unk_token: str = DEFAULT_UNK,
    ) -> None:
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        if vocab is not None:
            self.vocab = dict(vocab)
        else:
            # Initialize with default special tokens
            self.vocab = {
                self.pad_token: 0,
                self.bos_token: 1,
                self.eos_token: 2,
                self.unk_token: 3,
            }

        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}

        self.merges: List[Tuple[str, str]] = [tuple(m) for m in merges] if merges else []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {
            m: idx for idx, m in enumerate(self.merges)
        }

        # Cache for chunk tokenization to make inference ultra-fast
        self._cache: Dict[str, Tuple[str, ...]] = {}

    @property
    def pad_token_id(self) -> int:
        return self.vocab.get(self.pad_token, 0)

    @property
    def bos_token_id(self) -> int:
        return self.vocab.get(self.bos_token, 1)

    @property
    def eos_token_id(self) -> int:
        return self.vocab.get(self.eos_token, 2)

    @property
    def unk_token_id(self) -> int:
        return self.vocab.get(self.unk_token, 3)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def special_tokens(self) -> List[str]:
        return [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

    @property
    def special_token_ids(self) -> List[int]:
        return [self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id]

    def __len__(self) -> int:
        return self.vocab_size

    def get_vocab(self) -> Dict[str, int]:
        """Returns a copy of the vocabulary mapping."""
        return dict(self.vocab)

    def token_to_id(self, token: str) -> int:
        """Maps a token string to its integer ID (falls back to unk_token_id)."""
        return self.vocab.get(token, self.unk_token_id)

    def id_to_token_str(self, token_id: int) -> str:
        """Maps a token ID to its string representation (falls back to unk_token)."""
        return self.id_to_token.get(token_id, self.unk_token)

    def train(
        self,
        texts: Sequence[str],
        vocab_size: int,
        min_freq: int = 1,
        verbose: bool = False,
    ) -> "BPETokenizer":
        """
        Trains character and subword vocabulary and learns merge rules from a text corpus.

        Args:
            texts: Sequence of training strings.
            vocab_size: Desired final vocabulary size (must be >= 4).
            min_freq: Minimum frequency of a pair to be merged (default: 1).
            verbose: If True, prints progress details.

        Returns:
            self: The trained BPETokenizer instance.
        """
        if vocab_size < 4:
            raise ValueError(
                f"vocab_size must be at least 4 to accommodate special tokens, got {vocab_size}."
            )

        # 1. Reset vocabulary and merges
        self.vocab = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
        }
        self.merges = []
        self.bpe_ranks = {}
        self._cache.clear()

        # 2. Pre-tokenize corpus into chunks and compute chunk frequencies
        chunk_freqs: Counter[str] = Counter()
        for text in texts:
            if not text:
                continue
            text_meta = text.replace(" ", _META_SPACE)
            chunks = _CHUNK_PATTERN.findall(text_meta)
            chunk_freqs.update(chunks)

        # 3. Collect unique characters and character frequencies
        char_counts: Counter[str] = Counter()
        for chunk, freq in chunk_freqs.items():
            for c in chunk:
                char_counts[c] += freq

        # Determine available budget for base characters
        # If the number of unique characters exceeds the budget, select top frequent
        max_base_chars = vocab_size - len(self.vocab)
        if len(char_counts) > max_base_chars:
            sorted_chars = [c for c, _ in char_counts.most_common(max_base_chars)]
        else:
            # Sort for deterministic vocabulary order
            sorted_chars = sorted(char_counts.keys())

        for c in sorted_chars:
            if c not in self.vocab:
                self.vocab[c] = len(self.vocab)

        base_chars_set = set(self.vocab.keys())

        # 4. Initialize words as tuples of base tokens
        words: Dict[Tuple[str, ...], int] = {}
        for chunk, freq in chunk_freqs.items():
            word_tokens = tuple(c if c in base_chars_set else self.unk_token for c in chunk)
            words[word_tokens] = words.get(word_tokens, 0) + freq

        # 5. Iterative BPE merge learning
        num_merges_target = vocab_size - len(self.vocab)

        for merge_idx in range(num_merges_target):
            pairs: Dict[Tuple[str, str], int] = defaultdict(int)
            for word, freq in words.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pairs[pair] += freq

            if not pairs:
                break

            # Select pair with highest frequency; break ties deterministically
            best_pair = max(pairs.keys(), key=lambda p: (pairs[p], -len(p[0]) - len(p[1]), p))
            best_freq = pairs[best_pair]

            if best_freq < min_freq:
                break

            new_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.bpe_ranks[best_pair] = merge_idx
            self.vocab[new_token] = len(self.vocab)

            # Apply merge to all words in corpus
            new_words: Dict[Tuple[str, ...], int] = {}
            for word, freq in words.items():
                new_word: List[str] = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                t_word = tuple(new_word)
                new_words[t_word] = new_words.get(t_word, 0) + freq
            words = new_words

            if verbose and (merge_idx + 1) % 100 == 0:
                print(f"Merge {merge_idx + 1}/{num_merges_target}: {best_pair} -> '{new_token}' (freq={best_freq})")

        # Rebuild id_to_token map
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        return self

    def train_from_corpus(
        self,
        texts: Sequence[str],
        vocab_size: int,
        min_frequency: int = 1,
        verbose: bool = False,
    ) -> "BPETokenizer":
        """Convenience alias for train()."""
        return self.train(
            texts=texts,
            vocab_size=vocab_size,
            min_freq=min_frequency,
            verbose=verbose,
        )

    def _bpe_chunk(self, chunk: str) -> Tuple[str, ...]:
        """
        Applies learned BPE merge rules to a single pre-tokenized chunk.
        Uses LRU-style dictionary caching.
        """
        if chunk in self._cache:
            return self._cache[chunk]

        # Break into base tokens (characters in vocab or unk)
        tokens = [c if c in self.vocab else self.unk_token for c in chunk]
        if len(tokens) <= 1:
            res = tuple(tokens)
            self._cache[chunk] = res
            return res

        while len(tokens) > 1:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            candidate_pairs = [p for p in pairs if p in self.bpe_ranks]
            if not candidate_pairs:
                break

            # Find pair with lowest merge rank (learned earliest)
            best_pair = min(candidate_pairs, key=lambda p: self.bpe_ranks[p])

            new_tokens: List[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        res = tuple(tokens)
        self._cache[chunk] = res
        return res

    def tokenize(self, text: str, add_special_tokens: bool = False) -> List[str]:
        """
        Splits input text into subword token strings.

        Args:
            text: Text to tokenize.
            add_special_tokens: If True, prepends bos_token and appends eos_token.

        Returns:
            List of token strings.
        """
        if not text:
            tokens: List[str] = []
        else:
            text_meta = text.replace(" ", _META_SPACE)
            chunks = _CHUNK_PATTERN.findall(text_meta)
            tokens = []
            for chunk in chunks:
                tokens.extend(self._bpe_chunk(chunk))

        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encodes a string into a list of integer token IDs.

        Args:
            text: String to encode.
            add_special_tokens: Whether to include <bos> and <eos> tokens.

        Returns:
            List of integer token IDs.
        """
        tokens = self.tokenize(text, add_special_tokens=add_special_tokens)
        return [self.vocab.get(tok, self.unk_token_id) for tok in tokens]

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """
        Decodes a sequence of token IDs back into a string.

        Args:
            token_ids: Sequence of integer token IDs.
            skip_special_tokens: If True, removes <pad>, <bos>, and <eos> tokens.
            clean_up_tokenization_spaces: Reserved for API compatibility.

        Returns:
            The reconstructed string.
        """
        special_to_skip = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        tokens: List[str] = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_to_skip:
                continue
            token_str = self.id_to_token.get(tid, self.unk_token)
            tokens.append(token_str)

        text = "".join(tokens).replace(_META_SPACE, " ")
        return text

    def batch_encode(
        self,
        texts: Sequence[str],
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
        add_special_tokens: bool = True,
    ) -> BatchEncoding:
        """
        PyTorch-compatible batch encoding helper.

        Args:
            texts: List of strings to encode.
            padding: Padding strategy:
                     - True or "longest": pad to longest sequence in the batch.
                     - "max_length": pad to max_length (max_length must be provided).
                     - False: do not pad.
            max_length: Maximum sequence length for truncation / padding.
            truncation: If True, truncate sequences longer than max_length.
            return_tensors: "pt" to return PyTorch tensors, None or "list" for lists.
            add_special_tokens: Whether to prepend <bos> and append <eos>.

        Returns:
            BatchEncoding containing "input_ids" and "attention_mask".
        """
        raw_ids: List[List[int]] = [
            self.encode(text, add_special_tokens=add_special_tokens) for text in texts
        ]

        # Apply truncation if requested
        if truncation and max_length is not None:
            processed_ids = [ids[:max_length] for ids in raw_ids]
        else:
            processed_ids = raw_ids

        # Determine target length for padding
        if padding == "max_length":
            if max_length is None:
                raise ValueError("max_length must be specified when padding='max_length'")
            target_len = max_length
        elif padding is True or padding == "longest":
            target_len = max((len(ids) for ids in processed_ids), default=0)
            if max_length is not None and not truncation:
                target_len = max(target_len, max_length)
        else:
            target_len = None

        if target_len is not None:
            padded_ids: List[List[int]] = []
            attention_masks: List[List[int]] = []

            for ids in processed_ids:
                curr_len = len(ids)
                if curr_len < target_len:
                    pad_count = target_len - curr_len
                    padded_ids.append(ids + [self.pad_token_id] * pad_count)
                    attention_masks.append([1] * curr_len + [0] * pad_count)
                else:
                    padded_ids.append(ids[:target_len])
                    attention_masks.append([1] * target_len)
        else:
            padded_ids = processed_ids
            attention_masks = [[1] * len(ids) for ids in processed_ids]

        if return_tensors == "pt":
            if torch is None:
                raise ImportError(
                    "PyTorch is not installed or available. Cannot return torch.Tensor. "
                    "Install torch or set return_tensors=None."
                )
            input_ids_tensor = torch.tensor(padded_ids, dtype=torch.long)
            attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long)
            return BatchEncoding(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
            )

        return BatchEncoding(
            input_ids=padded_ids,
            attention_mask=attention_masks,
        )

    def encode_batch(
        self,
        texts: Sequence[str],
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
        add_special_tokens: bool = True,
    ) -> BatchEncoding:
        """Alias for batch_encode."""
        return self.batch_encode(
            texts=texts,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
        )

    def __call__(
        self,
        text_or_texts: Union[str, Sequence[str]],
        **kwargs: Any,
    ) -> BatchEncoding:
        """
        Convenience call operator (compatible with HuggingFace tokenizers).
        """
        if isinstance(text_or_texts, str):
            return self.batch_encode([text_or_texts], **kwargs)
        return self.batch_encode(text_or_texts, **kwargs)

    def save(self, path: Union[str, Path]) -> str:
        """
        Serializes the tokenizer configuration, vocabulary, and merges to a JSON file.

        Args:
            path: Target file path or directory (if directory, 'tokenizer.json' is created).

        Returns:
            The saved file path as string.
        """
        target = Path(path)
        if target.is_dir() or target.suffix == "":
            target.mkdir(parents=True, exist_ok=True)
            target = target / "tokenizer.json"
        else:
            target.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "type": "bpe",
            "special_tokens": {
                "pad_token": self.pad_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "unk_token": self.unk_token,
            },
            "vocab": self.vocab,
            "merges": self.merges,
        }

        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return str(target)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        """
        Loads a BPETokenizer instance from a JSON file or directory containing tokenizer.json.

        Args:
            path: Path to the JSON file or directory.

        Returns:
            A new BPETokenizer instance restored from the file.
        """
        target = Path(path)
        if target.is_dir():
            target = target / "tokenizer.json"

        if not target.exists():
            raise FileNotFoundError(f"Tokenizer file not found at: {target}")

        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)

        special = data.get("special_tokens", {})
        pad_token = special.get("pad_token", cls.DEFAULT_PAD)
        bos_token = special.get("bos_token", cls.DEFAULT_BOS)
        eos_token = special.get("eos_token", cls.DEFAULT_EOS)
        unk_token = special.get("unk_token", cls.DEFAULT_UNK)

        vocab = data.get("vocab", {})
        merges = [tuple(m) for m in data.get("merges", [])]

        return cls(
            vocab=vocab,
            merges=merges,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "BPETokenizer":
        """Alias for load."""
        return cls.load(path)
