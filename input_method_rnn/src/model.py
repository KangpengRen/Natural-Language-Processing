"""
模型
"""
import torch.nn as nn

from config import EMBEDDING_DIM, HIDDEN_SIZE


class InputMethodModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_DIM)  # 词嵌入层
        self.rnn = nn.RNN(input_size=EMBEDDING_DIM,
                          hidden_size=HIDDEN_SIZE,
                          batch_first=True,
                          num_layers=1,
                          bidirectional=False)  # RNN层
        self.linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=vocab_size)  # 线性层

    def forward(self, x):
        # x.shape = (batch_size, seq_len)
        embed = self.embedding(x)
        # embed.shape = (batch_size, seq_len, embedding_dim)
        output, _ = self.rnn(embed)  # 要求输入形状：(batch_size, seq_len, input_size)
        # output.shape = (batch_size, seq_len, hidden_size * num_layers)
        last_hidden_state = output[:, -1, :]  # 获取句子最后的处理结果
        # last_hidden_state = (batch_size, hidden_size)
        output = self.linear(last_hidden_state)
        # output.shape = (batch_size, vocab_size)
        return output
