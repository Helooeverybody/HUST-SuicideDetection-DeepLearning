import torch.nn as nn
import torch
import torch.nn.functional as F 
import torch.optim as optim

class SentimentLSTM(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_size, tagset_size, n_layers, dropout_rate = 0.3, pretrained_embeddings = None):
        super(SentimentLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embs = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            pretrained_embeddings = torch.tensor(pretrained_embeddings, dtype=torch.float32)
            self.embs.weight = nn.Parameter(pretrained_embeddings)
            self.embs.weight.requires_grad = False  # Freeze embedding weights if you don't want to train them

        self.lstm = nn.LSTM(embedding_dim, hidden_size,n_layers,dropout=dropout_rate, batch_first= True)
        self.hidden2tag = nn.Linear(hidden_size, tagset_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.sig = nn.Sigmoid()

    def forward(self, sentence):
        batch_size = sentence.size(0)
        embs = self.embs(sentence)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(embs.device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(embs.device)
        lstm_out, _ = self.lstm(embs.view(batch_size,embs.size(1), -1), (h_0,c_0))
        out = self.dropout(lstm_out)
        out = self.hidden2tag(out)
        out = out[:,-1,:]
        sig_out = self.sig(out)
        return sig_out.squeeze()



