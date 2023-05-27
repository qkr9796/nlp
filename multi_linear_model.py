from transformers import AutoTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelMultiLinear(nn.Module):
    
    def __init__(self, model_name, num_intent_classes, num_tag_classes):
        super(ModelMultiLinear, self).__init__()
        
        d = 1024
        
        self.bert = BertModel.from_pretrained(model_name)
        self.config = BertConfig.from_pretrained(model_name)
        
        #self.intent_classifier = nn.Linear(in_features=self.config.hidden_size, out_features=num_intent_classes)
        
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
        
        #self.tag_classifier = nn.Linear(in_features=self.config.hidden_size, out_features=num_tag_classes)
        
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
        #intent = F.sigmoid(intent)

        tag = self.tag_classifier(output.last_hidden_state[:,1:])

        #tag = F.sigmoid(tag)
        
        return intent, tag