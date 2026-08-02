from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import multiprocessing as mp

from collections import Counter

import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_tokens, list), "Must represent special tokens as a list of bytestrings"
    assert all(isinstance(token, bytes) for token in split_special_tokens), "All special tokens must be bytestrings"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            pattern = re.compile(b'|'.join([re.escape(token) for token in split_special_tokens]))
            found_at = pattern.search(mini_chunk)
            if found_at is not None:
                chunk_boundaries[bi] = initial_position + found_at.start()
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenization(chunk, special_tokens):
    counter = Counter()
    for chunk_small in re.split('|'.join([re.escape(token) for token in special_tokens]), chunk):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for match in re.finditer(PAT, chunk_small):
            if len(chunk_small) < 1:
                print("match:", match)
            res = match.group().encode("utf8")
            tmp = tuple([res[i: i+1] for i in range(len(res))])
            counter[tmp] += 1
    return counter


def initialize_vocab(special_tokens_bytes: list[bytes]) -> dict[int, bytes]:
    vocab = {}
    for i, special_token in enumerate(special_tokens_bytes):
        vocab[i] = special_token
    
    for i in range(256):
        vocab[len(vocab)] = bytes([i])
    return vocab


def BPE_update(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], counter, counter_tuple: Counter | None):
    if counter_tuple is None:
        counter_tuple = Counter()
        for tp in counter:
            num = counter[tp]
            for i in range(len(tp) - 1):
                counter_tuple[(tp[i], tp[i + 1])] += num
    
    key_idx = None
    for key in counter_tuple.keys():
        if key_idx is None or (counter_tuple[key], key[0], key[1]) > (counter_tuple[key_idx], key_idx[0], key_idx[1]):
            key_idx = key
    
    new_token_1 = key_idx[0]
    new_token_2 = key_idx[1]
    new_token = new_token_1 + new_token_2

    vocab[len(vocab)] = new_token
    merges.append((new_token_1, new_token_2))
    
    l = list(counter.keys())
    
    for key in l:
        tmp = []
        i = 0
        while i < len(key):
            if i + 1 < len(key) and key[i] == new_token_1 and key[i + 1] == new_token_2:
                tmp.append(new_token_1 + new_token_2)
                i += 2
            else:
                tmp.append(key[i])
                i += 1
        tmp = tuple(tmp)
        if key != tmp:
            count = counter.pop(key)
            counter[tmp] = count
            for i in range(len(tmp) - 1):
                counter_tuple[(tmp[i], tmp[i + 1])] += count
            for i in range(len(key) - 1):
                counter_tuple[(key[i], key[i + 1])] -= count

    return vocab, merges, counter, counter_tuple


def counter_file_chunk(input_path, start, end, special_tokens):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        counter = pre_tokenization(chunk, special_tokens)
        f.close()

    return counter


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    special_tokens_byte = [token.encode("utf-8") for token in special_tokens]
    vocab = initialize_vocab(special_tokens_byte)

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens_byte)
        f.close()

    num_processes = len(boundaries) - 1

    tasks = []

    for i in range(num_processes):
        start = boundaries[i]
        end = boundaries[i + 1]
        tasks.append([input_path, start, end, special_tokens])
    
    with mp.Pool(processes = num_processes) as pool:
        partial_counters = pool.starmap(counter_file_chunk, tasks)

    counter = Counter()
    for p_c in partial_counters:
        counter.update(p_c)
    
    merges = []
    
    total = vocab_size - len(vocab)
    
    counter_tuple = None
    for _ in range(total):
        vocab, merges, counter, counter_tuple = BPE_update(vocab, merges, counter, counter_tuple)
    
    return (vocab, merges)
