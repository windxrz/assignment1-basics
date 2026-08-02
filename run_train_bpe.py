import json
import time

import pickle as pkl

import os

from tests.adapters import run_train_bpe


def train_bpe(dataset, vocab_size=1000):    
    output_filename = os.path.join("output", f"{dataset}_vocab_size_{vocab_size}_BPE.pkl")
    
    if not os.path.exists(output_filename):
    # if True:
        input_path = os.path.join("data", dataset)
        start_time = time.time()
        vocab, merges = run_train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
        )
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time for dataset {dataset} is", end_time - start_time)
        with open(output_filename, "wb") as f:
            pkl.dump({
                "time": training_time,
                "vocab": vocab,
                "merges": merges
            }, f)
            f.close()
    else:
        with open(output_filename, "rb") as f:
            res = pkl.loads(f.read())
            vocab = res["vocab"]
            merges = res["merges"]
            training_time = res["time"]
            f.close()
    return vocab, merges, training_time


if __name__ == "__main__":
    vocab, merges, training_time = train_bpe("TinyStoriesV2-GPT4-train.txt", 10000)
    tmp = [(-len(ele), ele) for ele in vocab.values()]
    tmp = sorted(tmp)
    print(tmp[:10])
