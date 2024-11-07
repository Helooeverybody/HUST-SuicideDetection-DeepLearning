import torch 

class PretrainData(torch.utils.data.Dataset): 
    def __init__(self, encodings): 
        self.encodings = encodings
    def __getitem__(self,idx):
        return {key: val[idx] for key, val in  self.encodings.items()}
    def __len__(self): 
        return len(self.encodings)

class FinetuneData(torch.utils.data.Dataset): 
    def __init__(self,encodings):
        self.encodings = encodings
    def __getitem__(self,idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings)

    