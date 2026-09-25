import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Block


corpus = [
    "hello friends how are you",
    "the tea is very hot",
    "my name is Sabbir",
    "the roads of Dhaka are busy",
    "it is raining in Chittagong",
    "the train is late again",
    "i love eating samosas and drinking tea",
    "Pohela Boishakis my favorite festival",
    "Eid brings lights and sweets",
    "Bangladesh won the cricket match"
]

corpus = [s + " <END>" for s in corpus]
text = " ".join(corpus)
print(text)

words = list(set(text.split()))
print(words)

vocab_size = len(words)
print(f"Vocabulary size: {vocab_size}")

word_to_idx = {word: idx for idx, word in enumerate(words)}
print(word_to_idx)

word_to_idx = {idx: word for word, idx in enumerate(words)}
print(word_to_idx)

data = torch.tensor([word_to_idx[word] for word in text.split()], dtype=torch.long)
print(data)

print(f"Data shape: {data.shape}")

block_size = 6
embed_size = 32
n_heads = 2
n_layers = 2
lr = 1e-3
epochs = 1500