import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM_Attention(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_size, tagset_size, n_layers, dropout_rate = 0.3, pretrained_embeddings = None):
        super(BiLSTM_Attention,self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embs = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            pretrained_embeddings = torch.tensor(pretrained_embeddings, dtype=torch.float32)
            self.embs.weight = nn.Parameter(pretrained_embeddings)
            self.embs.weight.requires_grad = False  

        self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers,dropout=dropout_rate, bidirectional= True,batch_first= True)
        self.hidden2tag = nn.Linear(hidden_size*2, tagset_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.sig = nn.Sigmoid()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size*2 , 1),
            nn.ReLU(inplace = True)
        )
    def attention_layer(self, lstm_output):
        attn_weights = torch.tanh(self.attention(lstm_output)) # batch, seq, 1
        attn_weights = attn_weights.squeeze(2)
        attn_weights = F.softmax(attn_weights, dim = 1)

        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output) # batch,1, hidden*2
        context = context.squeeze(1) 
        return context, attn_weights
    
    def forward(self, sentence):
        batch_size = sentence.size(0)
        embs = self.embs(sentence)
        h_0 = torch.zeros(self.n_layers *2, batch_size, self.hidden_size).to(embs.device)
        c_0 = torch.zeros(self.n_layers *2, batch_size, self.hidden_size).to(embs.device)
        lstm_out, _ = self.lstm(embs.view(batch_size,embs.size(1), -1), (h_0,c_0))
        context, attn_weights = self.attention_layer(lstm_out)
        out = self.dropout(context)
        out = self.hidden2tag(out)
        tag_scores = self.sig(out)
        return tag_scores.squeeze()



