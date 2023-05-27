from transformers import AutoTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    
    def __init__(self, num_intent_classes, num_tag_classes, num_hidden_layers=3):
        super(Model, self).__init__()
        
        d = 1024
        
        self.config = BertConfig(num_hidden_layers=num_hidden_layers)
        self.bert = BertModel(self.config)
        
        self.intent_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, d),
            nn.LayerNorm(d),
            nn.LeakyReLU(),
            nn.Linear(d, d*2),
            nn.LayerNorm(d*2),
            nn.LeakyReLU(),
            nn.Linear(d*2, d*4),
            nn.LayerNorm(d*4),
            nn.LeakyReLU(),
            nn.Linear(d*4, num_intent_classes))
        
        self.tag_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, d),
            nn.LayerNorm(d),
            nn.LeakyReLU(),
            nn.Linear(d, d*2),
            nn.LayerNorm(d*2),
            nn.LeakyReLU(),
            nn.Linear(d*2, d*4),
            nn.LayerNorm(d*4),
            nn.LeakyReLU(),
            nn.Linear(d*4, num_tag_classes))
                
        
       
    def forward(self,inputs):
        output = self.bert(**inputs)
        intent = self.intent_classifier(output.last_hidden_state[:,0])

        tag = self.tag_classifier(output.last_hidden_state[:,1:])
        
        return intent, tag