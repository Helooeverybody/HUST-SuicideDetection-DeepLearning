import torch 
from torch import nn 
from bert_layer import Bert_Layer


class Bert_Pretrain(nn.Module):
    def __init__(self,config):
        super(Bert_Pretrain,self).__init__()
        # bert encode
        self.bert_encode = Bert_Layer(config.vocab_size,
                                    config.drop_out,
                                    config.dim_inp, 
                                    config.dim_out,
                                    config.dim_intermediate,
                                    config.num_heads, 
                                    config.num_layers)
        # output of bert pretrain
        self.bert_pretrain_out = nn.Sequential(nn.Linear(in_features= config.dim_intermediate , out_features = config.vocab_size),
                                        nn.GELU(),
                                        nn.LayerNorm(config.vocab_size))
                    
    
    def forward(self,input_token: torch.tensor, mask: torch.tensor = None):
        output = self.bert_encode(input_token, mask)
        output = self.bert_pretrain_out(output)
        return output.mean(dim = 1)


    

