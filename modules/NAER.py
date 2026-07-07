import torch
import torch.nn as nn
from .BERT import Bert
from .ViTrans import PretrainedViT
from .MCTrans import MultiScaleEvidenceAttentionModule
from .EAAS import EvidenceAwareAttentionShiftMixingModule
from .NADER import NADERClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(config.dim_model, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        self.bert = Bert(config)
        self.vitrans = PretrainedViT(model_path=config.vit_path)
        self.easm = EvidenceAwareAttentionShiftMixingModule(
            dim=768,
            num_heads=8,
            text_scales=(1, 3, 5),
            image_scales=(1, 3, 5),
            image_grid_size=(14, 14),
            shift_hidden_dim=128,
            evidence_hidden_dim=256,
            mix_hidden_dim=256,
            ffn_hidden_dim=1024,
            attn_dropout=0.1,
            proj_dropout=0.1,
            ffn_dropout=0.1,
            tau_text=1.0,
            tau_image=1.0,
        )

        self.nader = NADERClassifier(
        dim=768,
        num_heads=8,
        num_classes=3,
        dropout=0.1
    )


    def vit_encoder(self, x):
        features = self.vitrans(x)
        cls = features['cls_token']
        img = features['img_token']
        return img, cls.squeeze(1)


    def forward(self, x):
        img = x[0]
        context = x[1]  # 输入的句子
        mask = x[3]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        text, text_cls = self.bert(context, mask)
        img, img_cls = self.vit_encoder(img)
        text_evidence, image_evidence = self.easm(text, img, return_aux=False)
        logits, aux = self.nader(text_evidence, image_evidence)
        return logits