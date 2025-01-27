import time
from collections import Counter


class BPETokenizer:
    SEPARATOR_TOKEN = "<SEPARATOR>"

    def __init__(self, max_vocab=1000, enable_logs=True):
        self.vocab = None
        self.max_vocab = max_vocab
        self.enable_logs = enable_logs

    def _log_step(self, i, tokenized_text, elapsed_time):
        if self.enable_logs:
            print(f"{i}; last token '{self.vocab[-1]}'; |text|={len(tokenized_text)} tokens; |vocab|={len(self.vocab)} tokens; {elapsed_time}s")

    def fit(self, docs):        
        text = f" {self.SEPARATOR_TOKEN} ".join([d for d in docs])
        self.vocab = [self.SEPARATOR_TOKEN] + list(sorted(set(text)))
        tokenized_text = [(self.vocab.index(c), ) for c in text]
        
        i = 0
        time_point = time.time()
        elapsed_time = int(time.time() - time_point)
        self._log_step(i, tokenized_text, elapsed_time)
        time_point = time.time()

        while len(self.vocab) < self.max_vocab:
            c = Counter()
            for t1, t2 in zip(tokenized_text, tokenized_text[1:]):
                c[t1 + t2] += 1
            new_enc_token, new_enc_token_count  = c.most_common(1)[0]
            self.vocab.append("".join(self.vocab[i] for i in new_enc_token))

            idx = 0
            new_tokenized_text = []
            while idx < len(tokenized_text) - 1:
                if (tokenized_text[idx] + tokenized_text[idx+1]) == new_enc_token:
                    new_tokenized_text.append(new_enc_token)
                    idx += 2
                else:
                    new_tokenized_text.append(tokenized_text[idx])
                    idx += 1
            tokenized_text = new_tokenized_text

            i += 1
            elapsed_time = int(time.time() - time_point)
            self._log_step(i, tokenized_text, elapsed_time)
            time_point = time.time()

    def decode(self, tokens):
        return "".join(self.vocab[idx] for idx in tokens) 

    def encode(self, text):
        tokenized_text = list(text)

        indexed_tokens = [(0, self.SEPARATOR_TOKEN)]        
        for reverse_idx, token in enumerate(reversed(self.vocab[1:])):
            idx = len(self.vocab) - reverse_idx - 1
            indexed_tokens.append((idx, token))

        for idx, token in indexed_tokens:
            text_idx = 0
            new_tokenized_text = []
            while text_idx < len(tokenized_text):
                candidate = tokenized_text[text_idx:text_idx + len(token)]
                if all(isinstance(tt, str) for tt in candidate) and \
                   ("".join(candidate) == token):
                    new_tokenized_text.append(idx)
                    text_idx += len(token)
                else:
                    new_tokenized_text.append(tokenized_text[text_idx])
                    text_idx += 1
            tokenized_text = new_tokenized_text
        return tokenized_text
        