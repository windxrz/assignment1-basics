import os
import numpy as np

from cs336_basics.tokenizer import Tokenizer


def build_tokenizer(dataset, vocab_size):
    vocab_path = os.path.join("output", f"{dataset}_vocab_size_{vocab_size}_BPE.pkl")
    if not os.path.exists(vocab_path):
        return None
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer({}, [], special_tokens)
    tokenizer.from_files(vocab_path, vocab_path, special_tokens)
    return tokenizer


def a_encode_b(dataset1, dataset2, vocab_size):
    print("=" * 50)
    print(dataset1, " ==> ", dataset2)
    print("=" * 50)
    output_filename = os.path.join("output", f"{dataset1}_to_{dataset2}_tokenization.npy")
    if os.path.exists(output_filename):
        res = np.load(output_filename)
    else:
        tokenizer = build_tokenizer(dataset1, vocab_size)
        if tokenizer is None:
            return None
        data_path = os.path.join("data", dataset2)
        if not os.path.exists(data_path):
            return None
        res = []
        with open(data_path, "r") as f:
            for ele in tokenizer.encode_iterable(f):
                res.append(ele)
        res = np.array(res, dtype=np.uint16)
        np.save(output_filename, res)
    file_size = os.path.getsize(os.path.join("data", dataset2))
    token_size = len(res)
    print("compression ratio (bytes/token) = ", file_size / token_size)
    print("\n\n")
    return res


if __name__ == "__main__":
    a_encode_b("corpus.en", "corpus.en", 500)
    a_encode_b("TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-train.txt", 10000)
    a_encode_b("TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt", 10000)
    a_encode_b("owt_train.txt", "owt_train.txt", 32000)
    a_encode_b("owt_train.txt", "owt_valid.txt", 32000)
    a_encode_b("owt_train.txt", "TinyStoriesV2-GPT4-valid.txt", 32000)
    a_encode_b("TinyStoriesV2-GPT4-train.txt", "owt_valid.txt", 10000)
    a_encode_b("corpus.en", "TinyStoriesV2-GPT4-valid.txt", 500)
    a_encode_b("TinyStoriesV2-GPT4-train.txt", "corpus.en", 10000)
