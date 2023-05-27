from transformers import AutoTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelTwoLinear(nn.Module):
    
    def __init__(self, model_name, num_intent_classes, num_tag_classes):
        super(ModelTwoLinear, self).__init__()
        
        d = 2048
        
        self.bert = BertModel.from_pretrained(model_name)
        self.config = BertConfig.from_pretrained(model_name)
        
        self.intent_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, d),
            nn.Dropout(p=0.2),
            nn.Linear(d, num_intent_classes)
        )
        
        self.tag_classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, d),
            nn.Dropout(p=0.2),
            nn.Linear(d, num_tag_classes)
        )
        
        
       
    def forward(self,inputs):
        output = self.bert(**inputs)
        intent = self.intent_classifier(output.last_hidden_state[:,0])
        #intent = F.sigmoid(intent)

        tag = self.tag_classifier(output.last_hidden_state[:,1:])

        #tag = F.sigmoid(tag)
        
        return intent, tag