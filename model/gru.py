import torch
import torch.nn as nn
class GRU(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_size,tagset_size, n_layers, dropout_rate ,bid = False, pretrained_embeddings = None):
        super(GRU,self).__init__()
        self.vocab_size=vocab_size
        self.embedding_dim=embedding_dim
        self.hidden_size=hidden_size
        self.n_layers=n_layers
        self.tagset_size=tagset_size
        self.d=2 if bid else 1
        self.embs=nn.Embedding(vocab_size,embedding_dim)
        if pretrained_embeddings is not None:
            pretrained_embeddings = torch.tensor(pretrained_embeddings, dtype=torch.float32)
            self.embs.weight = nn.Parameter(pretrained_embeddings)
            self.embs.weight.requires_grad = False  # Freeze embedding weights if you don't want to train them
        self.gru=nn.GRU(embedding_dim,hidden_size,n_layers,dropout=dropout_rate,batch_first=True)
        self.fc=nn.Linear(hidden_size,tagset_size)
        self.sig=nn.Sigmoid()
    def init_hidden(self,x):
        self.batch_size = x.size(0)
        return torch.zeros((self.d*self.n_layers,self.batch_size,self.hidden_size)).cuda()
    def forward(self,x):
        x=self.embs(x)
        gru_out,final_hidden=self.gru(x,self.init_hidden(x)) # gru_out : (batch_size,sequence_length, hidden_size)
        gru_out=gru_out[:,-1,:] #(batch_size,hidden_size) extract the hidden state of the last timestep
        out=self.fc(gru_out) #(batch_size,1)
        out=self.sig(out) #(batch_size,1)
        return out.squeeze()