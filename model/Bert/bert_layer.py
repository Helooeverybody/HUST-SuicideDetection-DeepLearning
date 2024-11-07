import numpy as np 
import torch 
from torch import nn 
import torch.nn.functional as f  


class Total_Embedding(nn.Module):
    ''' total embedding class includes words embedding, segment embedding, and position encode'''
    def __init__(self,vocab_size : int,dim_out : int, dim_inp):
        super(Total_Embedding,self).__init__()
        self.token_embedding = nn.Embedding(vocab_size,dim_out)
        self.position_embedding = nn.Embedding(dim_inp,dim_out)
        self.register_buffer("position_ids", torch.arange(dim_inp).expand((1, -1)))
        self.norm = nn.LayerNorm(dim_out)

    def forward(self,input_token : torch.tensor) -> torch.tensor: 
        # length of input 
        seq_length = input_token.size(1)
        # take position 
        position_ids = self.position_ids[:, :seq_length].to(input_token.device)

        output = self.token_embedding(input_token) + self.position_embedding(position_ids)

        return self.norm(output)


class AttentionHead(nn.Module):
    def __init__(self,dim_inp : int ,dim_out: int):
        super(AttentionHead,self).__init__()

        self.Q = nn.Linear(dim_out,dim_out, bias = False)
        self.K = nn.Linear(dim_out,dim_out, bias = False)
        self.V = nn.Linear(dim_out,dim_out, bias = False)

    def forward(self,input_vector: torch.tensor, mask : torch.tensor = None) -> torch.tensor:
        ''' implement attention mechanism
            Detail:   calulus attention score:
                1. Each vector represent for a token, the vector linear mapping in three matrix query, key, and value. 
                2. Calculus attention score:   attention(Q,K,V) = softmax(QK^T/(d_k)^{0.5}).V '''

        q , k , v = self.Q(input_vector), self.K(input_vector), self.V(input_vector)
        
        scale = k.size(1)**(0.5)    
        scores = torch.matmul(q, k.transpose(-2,-1)) / scale 

        if mask != None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = f.softmax(scores, dim = -1)
        output = torch.matmul(scores,v)
        return output 

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads: int, dim_inp:int, dim_out: int):
        super(MultiHeadAttention,self).__init__()
        self.heads = nn.ModuleList([AttentionHead(dim_inp, dim_out) for _ in range(num_heads)])
        self.linear = nn.Linear(dim_out*num_heads, dim_out)
        self.norm = nn.LayerNorm(dim_out)

    def forward(self,input_vector, mask : torch.tensor = None):
        mult_attention = torch.concat([head(input_vector, mask) for head in self.heads], dim = -1)

        # after implement multi-head attention, the result will feed in to feedforward network 
        feedforward = self.linear(mult_attention)
        output = self.norm(feedforward)
        return output 

class Encoder(nn.Module):
    def __init__(self, num_heads, dim_inp, dim_out,dropout = 0.2):
        super(Encoder,self).__init__()
        self.attention = MultiHeadAttention(num_heads = num_heads, dim_inp = dim_inp, dim_out = dim_out)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_out,dim_out),
            nn.Dropout(p = dropout),
            nn.ReLU(),
            nn.Linear(dim_out,dim_out),
            nn.Dropout(p = dropout)
        )
        self.norm = nn.LayerNorm(dim_out)

    def forward(self,token_input, mask : torch.tensor = None):
        context = self.attention(token_input, mask)
        output = self.feed_forward(context)
        output = self.norm(output) + output
        return output 

class Bert_Layer(nn.Module):
    def __init__(self, vocab_size,drop_out = 0.1,dim_inp = 300, dim_out = 128, dim_intermediate = 256, num_heads = 8, num_layers = 1):
        super(Bert_Layer,self).__init__()
        self.num_layers = num_layers
        self.embedding = Total_Embedding(vocab_size=vocab_size, dim_out= dim_out, dim_inp= dim_inp)
        self.encoder = Encoder(num_heads= num_heads, dim_inp= dim_inp, dim_out= dim_out)
        self.intermediate = nn.Sequential(nn.Linear(in_features = dim_out, out_features = dim_intermediate),
                                        nn.GELU()
                                        )
    def forward(self,token_input: torch.tensor, mask_attention: torch.tensor = None): 
        # for fintune 
        embedded = self.embedding(token_input)
        output = torch.concat([self.encoder(embedded,mask_attention) for _ in range(self.num_layers)],dim = -1)
        output = self.intermediate(output)
        return output 

if __name__ == '__main__':
    # test class total embedding 
    if not True:
        embedding = Total_Embedding(vocab_size = 100 , dim_out = 25, dim_inp= 5) 
        pos = embedding(torch.tensor([[[23,14,5,7,8]]]))
        print(pos)

    # test attentionhead class 
    if not True:
        attn = AttentionHead(
            dim_inp= 5, dim_out= 10
        )
        output = attn(torch.tensor([[[1,2,3,4,5]]], dtype = torch.float))
        print(output)

    # test multi-head attention 
    if not True:
        mh_attn = MultiHeadAttention(
            num_heads= 8, dim_inp= 5, dim_out= 10 
        )

        output = mh_attn(torch.tensor([[[1,2,3,4,5]]], dtype = torch.float))
        print(output)

    # test Encoder
    if not True:
        bert_layer = Bert_Layer(vocab_size= 10,dim_inp=5, dim_out=10,num_heads=3)
        output = bert_layer(torch.tensor([[[1,2,3,4,5]]]), torch.tensor([[[1,0,1,1,1]]], dtype = torch.float))
        print(output)

        
