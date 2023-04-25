from typing import Optional
from transformers import PreTrainedModel, PretrainedConfig, DistilBertModel
import torch
from torch import nn

class TransformerBasedModelDistilBert(nn.Module):
    def __init__(self):
        super(TransformerBasedModelDistilBert, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.55)
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        input_shape = input_ids.size()
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
            
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits

class MyConfigDistil(PretrainedConfig):
    model_type = "distilbert"
    def __init__(self, final_dropout=0.55, **kwargs):
        super().__init__(**kwargs)
        self.final_dropout = final_dropout
        
class MyHFModel_DistilBertBased(PreTrainedModel):
    config_class = MyConfigDistil
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = TransformerBasedModelDistilBert()
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        input_shape = input_ids.size()
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        return self.model(input_ids=input_ids, attention_mask=attention_mask) 

config = MyConfigDistil(0.55)
HF_DistilBertBasedModelAppDocs = MyHFModel_DistilBertBased(config)
