import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
	print(f"GPU name: {torch.cuda.get_device_name(0)}")

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