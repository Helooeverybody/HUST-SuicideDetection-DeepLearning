import numpy as np 
import random 
import pandas as pd 
import torch 
from torch import nn as nn 
import torch.nn.functional as f 
from torchtext.data.utils import get_tokenizer
from  collections import Counter, OrderedDict
from torchtext.vocab import vocab 
import json  

''' Bert focus on two tasks on pretraining transformer  

# 1. Next Sentence prediction (NSP) 
NSP is a binary classification task. Having two sentences in input, our model  
should be able to predict if the second sentence is a true continuation of the first sentence. 

# 2. Masked Language Model (MLM)
MLM is the task to predict the hidden words in the sentence

 '''

class PreDataSet:
    CLS = '[CLS]'
    PAD = '[PAD]'
    SEP = '[SEP]'
    MASK = '[MASK]'
    UNK = '[UNK]'

    def __init__(self,config = None,dataset = None ,name_text = None):
        self.dataset = dataset
        self.name_text = name_text
        self.mask_percentage = config.mask_percentage

        # dataframe for pretraining 
        self.data_mask = {'mask_attention':[],
                        'mask_token':[],
                        'mask_target':[]
                        }
        self.data_nsp = {'nsp_token': [],
                        'nsp_target': []
                        }

        # dataframe for funetuing 
        self.data_finetune = {'review_token':[], 
                                'target':[]}

        self.tokenizer = get_tokenizer('basic_english') 
        self.optimal_sentence_length = config.dim_inp
        self.vocab = None 
        self.counter = Counter() 
    
    def _mapping_index(self,lst_token: list, mode = 'context') -> torch.tensor:
        if mode == 'target':
            target = torch.zeros(len(self.vocab))
    
            target[[self.vocab[word] for word in lst_token[0]]] = 1
            return target 
        return torch.tensor([self.vocab[word] for word in lst_token])
    
    def _mapping_query(self,query):
        '''processing query input of user to check suicide '''
        if self.vocab != None:
            with open('vocab.json', 'r') as file:    
                self.vocab = json.load(file)
    
        lst_token = self.tokenizer(query)
        mapping_token = self._mapping_index(lst_token)

    def _fill_vocab(self):
        # fill on vocab libary special tocken " [CLS], [PAD], [SEP], [MASK], [UNK]"
        self.vocab.insert_token(self.PAD,0)
        self.vocab.insert_token(self.MASK,1) 
        self.vocab.insert_token(self.CLS,2)
        self.vocab.insert_token(self.SEP,3) 
        self.vocab.insert_token(self.UNK,4)

        self.vocab.set_default_index(4)

    def _update_vocab(self, sentence):
        self.counter.update(sentence)
        ordered_dict = OrderedDict(self.counter.most_common())
        self.vocab = vocab(ordered_dict)

    def _make_pad(self,sentence_tk: list) -> list:
        '''making full equal length for all sentence  '''
        if len(sentence_tk) >= self.optimal_sentence_length:
            result = sentence_tk[:self.optimal_sentence_length]
        else:
            pad_len = ['[PAD]' for _ in range(self.optimal_sentence_length - len(sentence_tk))]
            result = sentence_tk + pad_len 
            
        return result 

    def _make_mask(self,sentence: list) -> pd.DataFrame:
        ''' Each sentence is masked about 15 percent by [MASK] tocken  '''
        
        sentence_tk = self.tokenizer(sentence)
        # sorted frequen token 
        
        mask_index  = random.choices(range(len(sentence_tk)), k = int(self.mask_percentage*len(sentence_tk)))

        # check if there is not any mask in sentence  
        if mask_index == []:
            return None 
        
        result = {'token_context':[],'mask_attention':[], 'target':[]}
        result['target'].append([sentence_tk[i] for i in mask_index])
        for idx in mask_index:
            sentence_tk[idx] = '[MASK]'   
        mask_atten = torch.ones(self.optimal_sentence_length)
        mask_atten[mask_index] = 0 
        result['mask_attention'].append(mask_atten)
        result['token_context'].append(sentence_tk) 
        
        return result
    
    def MLM_task(self, list_sentence: list):
        mlm_result = {'token_context': [],'mask_attention': [],'target':[]}

        for sentence in list_sentence:
            result = self._make_mask(sentence)
            if result == None:
                continue
            mlm_result['token_context'] += result['token_context']
            mlm_result['target'].append(result['target']) 
            mlm_result['mask_attention'] += result['mask_attention']
        
        return mlm_result 
    
    def NSP_task(self,list_sentence: list) -> pd.DataFrame:
        ''' creating positive and negative couples sentence in each context.
            
            Positive couples sentence :    Two consecutive sentences 
            Negative couples sentence :    Not two consecutive sentences 
            
            Adding two spectial tockens [CLS], [SEP] at the head and end of the first sentence, respectively
        '''
        
        a = len(list_sentence)
        list_sentence = [self.tokenizer(sentence) for sentence in list_sentence]

        positive = []
        negative = []
        num_negative = int(a/2) - 1 
         
        # postive nsp 
        for i in range(a-1):
            positive.append((list_sentence[i], list_sentence[i + 1]))
            
        # taking not consecutive sentence for negative 
        for _ in range(num_negative):
            index_rd = random.randint(0,a-1)
            distance = random.randint(0,a - index_rd - 1)
            negative.append((list_sentence[index_rd], list_sentence[index_rd + distance]))

        # adding special token [CLS], [SEP] 
        result = {'token_context': [], 'target':[]}
        for couple in positive:
            result['token_context'].append([self.CLS] + couple[0] + ['.'] + [self.SEP] + couple[1] + [self.SEP])  
      
            result['target'].append(torch.tensor([0,1]))
        for couple in negative:
            result['token_context'].append([self.CLS] + couple[0] + ['.'] + [self.SEP] + couple[1] + [self.SEP])

            result['target'].append(torch.tensor([1,0]))

        return result  

    def set_vocab(self):
        for i, paragraph in enumerate(self.dataset[self.name_text]):
            # processing for data funetuing 
            paragraph_token = self.tokenizer(paragraph)
            # update vocabulary 
            self._update_vocab(paragraph_token)
        self._fill_vocab()
        # save vocab 
        with open('vocab.json', 'w') as file:
            json.dump(self.vocab.get_stoi(), file)
        return len(self.vocab)
        
    def _finetune(self,vocab = None):
        if vocab == None:
            with open('vocab.json', 'r') as file:    
                self.vocab = json.load(file)
        else:
            self.vocab = vocab 
        
        for i, paragraph in enumerate(self.dataset[self.name_text]):
            paragraph_token = self.tokenizer(paragraph)

            # padding '[PAD]' 
            padding = self._make_pad(paragraph_token)

            # mapping index 
            mapping = self._mapping_index(padding)

            # target 
            target = torch.tensor([0,1]) if self.dataset['class'].iloc[i] == 'suicide' else torch.tensor([1,0])
            self.data_finetune['review_token'].append(mapping) 
            self.data_finetune['target'].append(target)


        return self.data_finetune
    
    def NextSentencePrediction(self, vocab = None) -> pd.Series:
        if vocab == None:
            with open('vocab.json', 'r') as file:    
                self.vocab = json.load(file)
        else:
            self.vocab = vocab 
        
        for i, paragraph in enumerate(self.dataset[self.name_text]):
            # split sentence 
            list_sentence = paragraph.split('.')
            list_sentence.pop(-1)

            nsp_token = self.NSP_task(list_sentence)

            self.data_nsp['nsp_token'] += nsp_token['token_context']
            self.data_nsp['nsp_target'] += nsp_token['target'] 

        data_nsp = pd.DataFrame(self.data_nsp)

        # tocken '[PAD]'
        data_nsp['nsp_token'] = data_nsp['nsp_token'].apply(lambda x : self._make_pad(x))

        # mapping index 
        data_nsp['nsp_token'] = data_nsp['nsp_token'].apply(lambda x: self._mapping_index(x))

        return data_nsp.to_dict()

    def MaskLanguageModel(self, vocab = None) -> pd.Series:
        if vocab == None:
            with open('vocab.json', 'r') as file:    
                self.vocab = json.load(file)
        else:
            self.vocab = vocab 
        
        for i, paragraph in enumerate(self.dataset[self.name_text]):
            # split sentence 
            list_sentence = paragraph.split('.')
            list_sentence.pop(-1)

            mask_token = self.MLM_task(list_sentence)

            # processing for data mask 
            self.data_mask['mask_token'] += mask_token['token_context']
            self.data_mask['mask_target'] += mask_token['target']
            self.data_mask['mask_attention'] += mask_token['mask_attention']

        data_mask = pd.DataFrame(self.data_mask)

        # tocken  '[PAD]'
        data_mask['mask_token'] = data_mask['mask_token'].apply(lambda x : self._make_pad(x))

        # mapping index 
        data_mask['mask_target'] = data_mask['mask_target'].apply(lambda x: self._mapping_index(x, mode = 'target'))
        data_mask['mask_token'] = data_mask['mask_token'].apply(lambda x: self._mapping_index(x))

        return data_mask.to_dict()


if __name__ == '__main__':
    # check the programming 
    path = 'C:/Machine learning/NLP/Project/Dataset/Suicide_Detection.csv'
    df = pd.read_csv(path)

    data = PreDataSet(dataset = df[:2], name_text= 'text')
    # if there is not vocab.json file, you must to setup vocab or adding a new vocab
    data.set_vocab()
    result = data.MaskLanguageModel()
    print(result.head())
