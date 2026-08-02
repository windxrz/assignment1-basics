import regex as re
from collections.abc import Iterable

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = None
        self.special_tokens_bytes = None
        if special_tokens is not None:
            special_tokens = sorted(special_tokens, key=lambda x: -len(x))
            self.special_tokens = special_tokens
            self.special_tokens_bytes = [ele.encode("utf8") for ele in special_tokens]
            vocab_list = vocab.values()
            for ele in self.special_tokens_bytes:
                if ele not in vocab_list:
                    self.vocab[len(self.vocab)] = ele

        self.vocab_inv = {}
        for k, v in self.vocab.items():
            self.vocab_inv[v] = k
        
        self.mini_chunk_size = 4096

    def _get_chunks(self, text: str, chunk_size: int = 40960):
        l = len(text)

        if self.special_tokens is None:
            return set((0, l))

        desired_num_chunks = l // chunk_size

        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = l

        mini_chunk_size = 4096

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            left = initial_position
            while True:
                mini_chunk = text[left: left + mini_chunk_size]

                if mini_chunk == "":
                    chunk_boundaries[bi] = l
                    break

                pattern = re.compile('|'.join([re.escape(token) for token in self.special_tokens]))
                found_at = pattern.search(mini_chunk)
                if found_at is not None:
                    chunk_boundaries[bi] = initial_position + found_at.start()
                    break
                initial_position += mini_chunk_size

        return sorted(set(chunk_boundaries))
    
    def from_files(self, vocab_filepath, merges_filepath, special_tokens=None):
        pass
    
    def _encode_word(self, word: str) -> list[int]:
        word_byte = word.encode("utf8")
        tmp = tuple([word_byte[i: i+1] for i in range(len(word_byte))])
        for (token1, token2) in self.merges:
            i = 0
            res = []
            while i < len(tmp):
                if i < len(tmp) - 1 and tmp[i] == token1 and tmp[i + 1] == token2:
                    res.append(tmp[i] + tmp[i + 1])
                    i += 2
                else:
                    res.append(tmp[i])
                    i += 1
            tmp = tuple(res)
        res = []
        for ele in tmp:
            res.append(self.vocab_inv[ele])
        return res

    def _encode_chunk(self, chunk: str) -> list[int]:
        res = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for m in re.finditer(PAT, chunk):
            word = m.group()
            res.extend(self._encode_word(word))
        return res

    def encode(self, text: str) -> list[int]:
        res = []
        res_special = []
        if self.special_tokens is not None:
            PAT_SPECIAL = '|'.join([re.escape(token) for token in self.special_tokens])
            chunks = []
            last = 0
            for m in re.finditer(PAT_SPECIAL, text):
                chunks.append(text[last: m.start()])
                res_special.append(self.vocab_inv[m.group().encode("utf8")])
                last = m.end()
            chunks.append(text[last: ])
        else:
            chunks = [text]
        for i, chunk in enumerate(chunks):
            res.extend(self._encode_chunk(chunk))
            if i < len(res_special):
                res.append(res_special[i])
        return res
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        str = ""
        for s in iterable:
            str += s
            if self.special_tokens is not None and len(str) % self.mini_chunk_size == 0:
                pattern = re.compile('|'.join([re.escape(token) for token in self.special_tokens]))
                found_at = pattern.search(str)
                if found_at is not None:
                    start = found_at.start()
                    end = found_at.end()
                    for ele in self._encode_chunk(str[:start]):
                        yield ele
                    yield(self.vocab_inv[str[start: end]])
                    str = str[end:]
        res = self.encode(str)
        for ele in res:
            yield ele

    def decode(self, ids: list[int]) -> str:
        res = b""
        for id in ids:
            res += self.vocab[id]
        res = res.decode("utf8", errors='replace')
        return res
