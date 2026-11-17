"""
Unit tests for BPETokenizer in scaleflow.tokenizer.bpe.
"""

import json
import os
import tempfile
import unittest

import torch

from scaleflow.tokenizer.bpe import BPETokenizer, BatchEncoding


class TestBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.sample_corpus = [
            "the quick brown fox jumps over the lazy dog",
            "the dog was not lazy at all",
            "recursive multiscale flow matching for image generation",
            "deep learning with byte pair encoding subword tokenization",
            "ultrafast diffusion flow models generate high resolution images",
            "hello world! testing special characters: 123, 456, and punctuation.",
        ]
        self.tokenizer = BPETokenizer()

    def test_default_special_tokens(self):
        self.assertEqual(self.tokenizer.pad_token, "<pad>")
        self.assertEqual(self.tokenizer.bos_token, "<bos>")
        self.assertEqual(self.tokenizer.eos_token, "<eos>")
        self.assertEqual(self.tokenizer.unk_token, "<unk>")

        self.assertEqual(self.tokenizer.pad_token_id, 0)
        self.assertEqual(self.tokenizer.bos_token_id, 1)
        self.assertEqual(self.tokenizer.eos_token_id, 2)
        self.assertEqual(self.tokenizer.unk_token_id, 3)

        self.assertEqual(len(self.tokenizer), 4)
        self.assertEqual(self.tokenizer.vocab_size, 4)

    def test_train_vocab_size(self):
        target_vocab_size = 80
        self.tokenizer.train(self.sample_corpus, vocab_size=target_vocab_size)

        self.assertEqual(self.tokenizer.vocab_size, target_vocab_size)
        self.assertEqual(len(self.tokenizer), target_vocab_size)
        self.assertTrue(len(self.tokenizer.merges) > 0)

        # Check special tokens remain at fixed IDs
        self.assertEqual(self.tokenizer.pad_token_id, 0)
        self.assertEqual(self.tokenizer.bos_token_id, 1)
        self.assertEqual(self.tokenizer.eos_token_id, 2)
        self.assertEqual(self.tokenizer.unk_token_id, 3)

    def test_train_invalid_vocab_size(self):
        with self.assertRaises(ValueError):
            self.tokenizer.train(self.sample_corpus, vocab_size=3)

    def test_encode_and_decode_reconstruction(self):
        self.tokenizer.train(self.sample_corpus, vocab_size=120)

        for text in self.sample_corpus:
            # Test without special tokens
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            self.assertIsInstance(ids, list)
            self.assertTrue(all(isinstance(i, int) for i in ids))

            decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
            self.assertEqual(decoded, text)

    def test_encode_with_special_tokens(self):
        self.tokenizer.train(self.sample_corpus, vocab_size=100)
        text = "the quick brown fox"
        ids = self.tokenizer.encode(text, add_special_tokens=True)

        self.assertEqual(ids[0], self.tokenizer.bos_token_id)
        self.assertEqual(ids[-1], self.tokenizer.eos_token_id)

        # Decoding with skip_special_tokens=True should recover exact original text
        decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
        self.assertEqual(decoded, text)

        # Decoding with skip_special_tokens=False should contain <bos> and <eos>
        decoded_raw = self.tokenizer.decode(ids, skip_special_tokens=False)
        self.assertTrue(decoded_raw.startswith("<bos>"))
        self.assertTrue(decoded_raw.endswith("<eos>"))

    def test_unknown_token_handling(self):
        # Train on simple ASCII only
        self.tokenizer.train(["cat dog"], vocab_size=15)
        text = "cat ⚡ dog"
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        self.assertIn(self.tokenizer.unk_token_id, ids)

        decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
        self.assertIn("<unk>", decoded)

    def test_batch_encode_pytorch(self):
        self.tokenizer.train(self.sample_corpus, vocab_size=100)
        texts = [
            "the quick brown fox",
            "the dog was not lazy at all and ran very fast",
            "short",
        ]

        # Test batch_encode with PyTorch tensors
        batch = self.tokenizer.batch_encode(
            texts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        self.assertIsInstance(batch, BatchEncoding)
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)

        input_ids = batch.input_ids
        attention_mask = batch.attention_mask

        self.assertIsInstance(input_ids, torch.Tensor)
        self.assertIsInstance(attention_mask, torch.Tensor)
        self.assertEqual(input_ids.dtype, torch.long)
        self.assertEqual(attention_mask.dtype, torch.long)

        # Dimensions should match (batch_size, max_seq_len)
        self.assertEqual(input_ids.shape[0], len(texts))
        self.assertEqual(input_ids.shape, attention_mask.shape)

        # Verify padding and mask consistency
        for b_idx in range(len(texts)):
            for seq_idx in range(input_ids.shape[1]):
                token_id = input_ids[b_idx, seq_idx].item()
                mask_val = attention_mask[b_idx, seq_idx].item()
                if token_id == self.tokenizer.pad_token_id:
                    self.assertEqual(mask_val, 0)
                else:
                    self.assertEqual(mask_val, 1)

    def test_batch_encode_with_max_length_and_truncation(self):
        self.tokenizer.train(self.sample_corpus, vocab_size=100)
        texts = [
            "the quick brown fox jumps over the lazy dog",
            "short text",
        ]

        max_len = 6
        batch = self.tokenizer.batch_encode(
            texts,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        )

        self.assertEqual(batch.input_ids.shape, (2, max_len))
        self.assertEqual(batch.attention_mask.shape, (2, max_len))

    def test_batch_encode_list_return(self):
        self.tokenizer.train(self.sample_corpus, vocab_size=100)
        texts = ["hello", "world of tokenizers"]

        batch = self.tokenizer.batch_encode(
            texts,
            padding=True,
            return_tensors="list",
        )

        self.assertIsInstance(batch["input_ids"], list)
        self.assertIsInstance(batch["attention_mask"], list)
        self.assertEqual(len(batch["input_ids"]), 2)
        self.assertEqual(len(batch["input_ids"][0]), len(batch["input_ids"][1]))

    def test_callable_interface(self):
        self.tokenizer.train(self.sample_corpus, vocab_size=100)

        # Single string input
        res1 = self.tokenizer("single prompt", return_tensors="pt")
        self.assertIsInstance(res1, BatchEncoding)
        self.assertEqual(res1.input_ids.shape[0], 1)

        # Multiple strings input
        res2 = self.tokenizer(["prompt one", "prompt two"], return_tensors="pt")
        self.assertEqual(res2.input_ids.shape[0], 2)

        # Device placement method test
        res_cpu = res2.to("cpu")
        self.assertEqual(res_cpu.input_ids.device.type, "cpu")

    def test_save_and_load_serialization(self):
        self.tokenizer.train(self.sample_corpus, vocab_size=90)
        test_text = "multiscale flow matching is ultrafast"
        original_ids = self.tokenizer.encode(test_text, add_special_tokens=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "tokenizer.json")
            self.tokenizer.save(save_path)

            self.assertTrue(os.path.exists(save_path))

            # Verify JSON structure
            with open(save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("vocab", data)
            self.assertIn("merges", data)
            self.assertIn("special_tokens", data)

            # Load into new instance
            loaded_tokenizer = BPETokenizer.load(save_path)
            self.assertEqual(len(loaded_tokenizer), len(self.tokenizer))
            self.assertEqual(loaded_tokenizer.vocab, self.tokenizer.vocab)
            self.assertEqual(loaded_tokenizer.merges, self.tokenizer.merges)

            loaded_ids = loaded_tokenizer.encode(test_text, add_special_tokens=True)
            self.assertEqual(loaded_ids, original_ids)

            decoded = loaded_tokenizer.decode(loaded_ids, skip_special_tokens=True)
            self.assertEqual(decoded, test_text)

    def test_edge_cases(self):
        self.tokenizer.train(self.sample_corpus, vocab_size=80)

        # Empty string
        empty_ids = self.tokenizer.encode("", add_special_tokens=False)
        self.assertEqual(empty_ids, [])
        self.assertEqual(self.tokenizer.decode(empty_ids), "")

        # Whitespace only
        space_ids = self.tokenizer.encode("   ", add_special_tokens=False)
        self.assertEqual(self.tokenizer.decode(space_ids), "   ")


if __name__ == "__main__":
    unittest.main()
