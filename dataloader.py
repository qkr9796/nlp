from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import torch


class SlurpTrainData(Dataset):
    
    def __init__(self, path, tokenizer):
        self.df = None
        self.type_labels = None
        self.intent_labels = None   
        
        self.df = pd.read_json(path + '/train.json')
        self.type_labels = pd.read_csv(path + '/type_labels.csv').iloc[:,0].to_numpy().tolist()
        self.intent_labels = pd.read_csv(path + '/intent_labels.csv').iloc[:,0].to_numpy().tolist()        
        
        self.df['intent'] = self.df['intent'].apply(self.intent_labels.index)
        
        self.y_label = self.df['intent']
        self.x = self.df['sentence']
        self.y = self.df['entities']
        
        self.tokenizer = tokenizer
        
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,idx):
        return self.x[idx], self.y_label[idx], self.y[idx]

class SlurpTestData(Dataset):
    def __init__(self, path, tokenizer):
        
        self.df = pd.read_json(path + '/test.json')
        self.type_labels = pd.read_csv(path + '/type_labels.csv').iloc[:,0].to_numpy().tolist()
        self.intent_labels = pd.read_csv(path + '/intent_labels.csv').iloc[:,0].to_numpy().tolist()
        
        self.df = self.df[self.df['intent'].isin(self.intent_labels)].reset_index() 
        self.df['intent'] = self.df['intent'].apply(self.intent_labels.index)
        
        
        self.y_label = self.df['intent']
        self.x = self.df['sentence']
        self.y = self.df['entities']
        
        self.tokenizer = tokenizer
        
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y_label[idx], self.y[idx]

class SlurpValidData(Dataset):
    def __init__(self, path, tokenizer):
        
        self.df = pd.read_json(path + '/valid.json')
        self.type_labels = pd.read_csv(path + '/type_labels.csv').iloc[:,0].to_numpy().tolist()
        self.intent_labels = pd.read_csv(path + '/intent_labels.csv').iloc[:,0].to_numpy().tolist()
        
        self.df = self.df[self.df['intent'].isin(self.intent_labels)].reset_index() 
        self.df['intent'] = self.df['intent'].apply(self.intent_labels.index)
        
        
        self.y_label = self.df['intent']
        self.x = self.df['sentence']
        self.y = self.df['entities']
        
        self.tokenizer = tokenizer
        
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y_label[idx], self.y[idx]
    

class SlurpCollate(object):
    
    def __init__(self, type_labels, tokenizer):
        
        self.type_labels = type_labels
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        
        x, y_label, y  = zip(*batch)   
                           
        out_x = self.tokenizer(x, padding=True, return_tensors='pt')
        out_y = [out_x.word_ids(i)[1:] for i in range(len(batch))]
        max_len = len(out_y[0])
        
        for idx in range(len(batch)):
            for k in range(max_len):
                if out_y[idx][k] == None:
                    out_y[idx][k] = -100
                else:
                    out_y[idx][k] = y[idx][out_y[idx][k]]

        
        y_label = torch.tensor(y_label)
        out_y = torch.tensor(out_y)
        
        return out_x, (y_label, out_y)  
