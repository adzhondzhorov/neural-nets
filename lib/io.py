from lib.tokenization import BPETokenizer


def save_tokenizer(tokenizer, name):
    with open(f"artifacts/tokenizers/{name}.tok", "wt") as f:
        f.write(str(tokenizer.max_vocab) + "\n")
        for token in tokenizer.vocab:
            f.write(token + "\n")

def load_tokenizer(name):
    with open(f"artifacts/tokenizers/{name}.tok", "rt") as f:
        lines = [t[:-1] for t in f.readlines()]
        tokenizer = BPETokenizer()
        tokenizer.max_vocab = int(lines[0])
        tokenizer.vocab = lines[1:]
        return tokenizer