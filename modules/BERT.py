import torch.nn as nn
from pytorch_pretrained import BertModel

class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(config.dropout)
        self.ln = nn.LayerNorm(config.dim_model)

    def forward(self, x, mask):
        out, cls_token = self.bert(x, attention_mask=mask, output_all_encoded_layers=False)
        return out, cls_token