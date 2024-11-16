import numpy as np 
import pandas as pd 
import torch 
from torch.optim import Adam 
from torch import nn 
import argparse
import json 
from tqdm import tqdm 
from bert_pretrain import Bert_Pretrain
from bert_finetune import Bert_Finetune
from dataset import PreDataSet
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from final_data import FinetuneData, PretrainData

device = ('cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available() else 'cpu')

def get_config():
    parser = argparse.ArgumentParser(description="BERT Model Config")
    
    # parameter for Bert 
    parser.add_argument('--vocab_size', type= int, default= None, help= "vocab size/ determinte by dataset process")
    parser.add_argument('--num_layers', type=int, default=1, help="numbers of layer bert")
    parser.add_argument('--dim_inp', type=int, default=300, help="max_length")
    parser.add_argument('--mask_percentage', type = int , default = 0.2, help = " the percentage of word in sentence masked")
    parser.add_argument('--dim_out', type=int, default=128, help="embedding size")
    parser.add_argument('--num_heads', type=int, default=8, help="num heads of attention")
    parser.add_argument('--dim_intermediate', type=int, default=256, help="dimension of intermediate layer")
    parser.add_argument('--drop_out', type=float, default=0.1, help="drop rate")
    parser.add_argument('--lr', type=float, default=0.002, help="learning rate")
    parser.add_argument('--epochs', type=int, default=100, help="number of epochs")
    parser.add_argument('--batch_size', type = int, default= 64,help = 'batch size')

    args = parser.parse_args()
    return args

def data_prepare(df : pd.DataFrame,vocab = None,config = None, mode = 'pretrain'):
    print('prepare data ---------------')
    
    # setup vocab 
    if vocab == None:
        vocab_size = PreDataSet(config,df, name_text= 'text').set_vocab()
        
    # setup data 
    if mode == 'pretrain':
        # dataset tool for pretrain 
        data_mask =  PreDataSet(config,df, name_text= 'text').MaskLanguageModel()

        data_mask = PretrainData(data_mask)
        train_loader = DataLoader(data_mask, batch_size = config.batch_size, shuffle = True)
        print('finish prepare data -------------')
        return train_loader, vocab_size
    else:
        # dataset tool for finetune 
        train_set , test_set = train_test_split(df, test_size=0.3)
        train_set =  PreDataSet(config,train_set, name_text= 'text')._finetune()
        test_set  =  PreDataSet(config,test_set, name_text= 'text')._finetune()

        train_set , test_set = FinetuneData(train_set), FinetuneData(test_set)

        train_loader = DataLoader(train_set, batch_size = config.batch_size , shuffle = True)
        test_loader = DataLoader(test_set, batch_size = config.batch_size , shuffle = True)

        print('finish prepare data------------- ')
        return train_loader, test_loader, vocab_size


class Trainer:
    def __init__(self, model, train_loader = None , test_loader = None , vocab = None, config = None):
        self.model = model
        self.train_loader = train_loader 
        self.test_loader = test_loader 
        self.loss_pretrain = nn.BCEWithLogitsLoss()
        self.loss_finetune = nn.CrossEntropyLoss()

        if vocab == None:
            with open('vocab.json', 'r') as file:
                self.vocab = json.load(file)
        else:
            self.vocab = vocab 
    
        if config != None:
            self.config = config
        else:
            self.config = get_config()
        self.logs = {'train_loss': [], 'val_loss': [], 'accuracy':[]}

    def pretrain_step(self):
        optimizer = Adam(self.model.parameters(), lr= self.config.lr)
        self.model.to(device)

        # train loop 
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0
            
            for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}"):
            
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass
                output = self.model(batch['mask_token'], batch['mask_attention'])
                loss = self.loss_pretrain(output,batch['mask_target'])
                train_loss += loss.item()

                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_train_loss = train_loss / len(self.train_loader)
            self.logs['train_loss'].append(avg_train_loss)
            print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")
        
        # save model 
        torch.save(self.model.state_dict(), 'bert-pretrained.pth')

    def finetune_step(self):
        optimizer = Adam(self.model.parameters(), lr= self.config.lr)
        self.model.to(device)

        # train loop 
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}"):
            
                batch = {k: v.to(device) for k, v in batch.items()}
            
                # Forward pass
                output = self.model(batch['review_token'])
                loss = self.loss_finetune(output,batch['target'].float())
                train_loss += loss.item()

                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_train_loss = train_loss / len(self.train_loader)
            self.logs['train_loss'].append(avg_train_loss)
            print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

            # evaluation 
            self.model.eval()
            val_loss = 0 
            correct = 0            
            with torch.no_grad():       
                for batch in self.test_loader:                                
                    batch = {k: v.to(device) for k, v in batch.items()}
                    output = self.model(batch['review_token'])
                    loss = self.loss_finetune(output,batch['target'].float())  
                    val_loss += loss.item()
                
                    # predict 
                    preds = torch.argmax(output, dim=1)
                    correct += (preds == batch['target']).sum().item()

                avg_val_loss = val_loss / len(self.test_loader)
                accuracy = correct / len(self.test_loader)
                self.logs['accuracy'].append(accuracy)
                self.logs['val_loss'].append(avg_val_loss)
                print(f"Epoch {epoch + 1} - Validation Loss: {avg_val_loss:.4f} - Accuracy: {accuracy:.4f}")
            
        torch.save(self.model.state_dict(), 'bert-finetune.pth')


if __name__ == '__main__':

    path = 'C:/Machine learning/NLP/Project/Dataset/Suicide_Detection.csv'
    df = pd.read_csv(path)
    config = get_config()
    # Example of setting pretrain and finetune

    # pretrain 
    if not True:
        # prepare data 
        train_loader, vocab_size = data_prepare(df = df[:1000],config = config,mode = 'pretrain')
        config.vocab_size = vocab_size
    
        # load model
        model = Bert_Pretrain(config)
        
        learner = Trainer(model,train_loader = train_loader)
        learner.pretrain_step()
    
    # finetune 
    if  True:
        # prepare data 
        train_loader, test_loader, vocab_size = data_prepare(df = df[:1000] , config= config, mode = 'finetune')
        config.vocab_size = vocab_size

        # load model  
        model = Bert_Finetune(config = config, model_pretrained_path= '/Machine learning/NLP/Project/Bert/bert-pretrained.pth')
    
        learner = Trainer(model, train_loader= train_loader, test_loader = test_loader)
        learner.finetune_step()

