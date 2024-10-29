import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, dropout=0.5, embedding_matrix=None):
        super(CNN, self).__init__()
        
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # for input of size (batch_size, sentence_length, embedding_dim)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        # -> (batch_size, 1, sentence_length, embedding_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # -> [(batch_size, num_filters, sentence_length - fs + 1) for fs in filter_sizes]
        x = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]
        # -> [(batch_size, num_filters) for fs in filter_sizes]
        x = torch.cat(x, 1)
        # -> (batch_size, num_filters * len(filter_sizes))
        x = self.dropout(x)
        x = self.fc(x)
        # -> (batch_size, 1)
        x = x.squeeze(1)
        # -> (batch_size)
        x = self.sigmoid(x)
        
        return x