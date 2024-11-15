import torch 
from torch import nn 
from bert_pretrain import Bert_Pretrain
from bert_layer import Bert_Layer


class Bert_Finetune(nn.Module):
    def __init__(self,config,labels = 2 ,model_pretrained_path = None):
        super(Bert_Finetune,self).__init__()
        if model_pretrained_path != None:
            # load the bert pretrained 
            self.model = Bert_Pretrain(config)
            self.model.load_state_dict(torch.load(model_pretrained_path))
            # remove last layer of bert pretrained 
            self.model.bert_pretrain_out = nn.Identity()
        else:
            self.model = Bert_Layer(config.vocab_size,
                                    config.drop_out,
                                    config.dim_inp, 
                                    config.dim_out,
                                    config.dim_intermediate,
                                    config.num_heads, 
                                    config.num_layers)
        # output layer of bert finetune 
        self.bert_finetune_out = nn.Linear(in_features= config.dim_intermediate, out_features= labels)    
                                            
    def forward(self, input_token: torch.tensor):
        
        output = self.model(input_token)
        output = self.bert_finetune_out(output)

        return output 




