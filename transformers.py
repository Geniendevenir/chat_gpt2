import numpy as np
import torch.nn as nn
from dataclasses import dataclass
from collections import OrderedDict

@dataclass
class GPT2Config:
	# Properties
	vocab_size: int = 50257 # 50000 + 256(ASCII) + 1 (END)
	embedding_size: int = 0 #Same as position encoding
	positional_encoding_dim: int = 0
 
	# Attention Properties
	attention_nb_head: int = 96
	context_block_size : int = 1024
	attention_block_nb : int = 4
	# K_dim
	# Q_dim
	# V_dim


class Transformer(nn.module):
	def __init__(self, config):
		super().__init__()
		self.transformer = nn.Sequential(OrderedDict(
		{
			# "Token" : nn.Linear(1, vocab_size), #Embedding pour token
			"Embedding" : nn.Embedding(config.vocab_size, config.embedding_size),
			"Positional encoding" : nn.Linear(config.context_block_size, config.embedding_size),
			"Block" :  [Block for _ in range(config.attention_block_nb)]
			"Linear" : nn.Linear(config.embedding_size, config.embedding_size),
		}
		))
		self.fnl = nn.Linear(config.embedding_size, config.embedding_size)
		# self.softmax = nn.softmax()


	def forward(self):
		

class Block(nn.module):
	def __init__(self, config):
		super().__init__()
		self.ln = nn.LayerNorm()
		self.atn = AttentionBlock
		
	def forward(self):


class AttentionBlock(nn.module):
	def __init__(self, config):
		super().__init__()
		self.nb_head = config.attention_nb_head
		self.KQV = nn.Linear(config.embedding_size, 3 * config.embedding_size)
		self.MLP = nn.Sequential(
			nn.Linear(config.embedding_size, 4 * config.embedding_size),
			nn.ReLu(),
			nn.Linear(4 * config.embedding_size, config.embedding_size))
	
	def forward(self, X):

		k, q, v = torch.split(X @ self.KQV)
		
		C = self.embedding_size // self.nb_head #Channels

		k = k.view(self.nb_head, C, self.embedding_size).reshape(-2, -1)
		q = q.view(self.nb_head, C, self.embedding_size).reshape(-2, -1)
		v = v.view(self.nb_head, C, self.embedding_size).reshape(-2, -1)

		atn = (k @ q.transpose(-2, -1) / torch.sqrt(self.embedding_size)) # (nb_head, embedding_size, embedding_size)
		mask = torch.tril(torch.ones(embedding_size, embedding_size), diagonal=-1)
		atn.masked_fill_(mask, float('-inf'))
		atn = torch.softmax(atn) @ v
		atn = atn.reshape(-2, -1).view(self.embedding_size, self.embedding_size)
		return (self.MLP(atn @ X))