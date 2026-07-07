import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import os

class PretrainedViT(nn.Module):
    def __init__(self, model_path="sylai/vit", freeze_params=True):
        """
        预训练ViT模块

        Args:
            model_path: 预训练模型参数路径
            freeze_params: 是否冻结ViT参数
        """
        super(PretrainedViT, self).__init__()

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise ValueError(f"模型路径不存在: {model_path}")

        # 加载ViT配置和模型
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            # 从本地加载配置
            config = ViTConfig.from_pretrained(model_path)
            self.vit = ViTModel(config)

            # 加载预训练权重
            model_file = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(model_file):
                state_dict = torch.load(model_file, map_location='cpu')
                self.vit.load_state_dict(state_dict)
                print(f"成功从 {model_path} 加载预训练权重")
            else:
                print(f"警告: 在 {model_path} 中未找到预训练权重文件")
        else:
            # 如果本地没有配置，尝试从transformers加载默认配置
            print("使用默认ViT配置")
            self.vit = ViTModel.from_pretrained(
                "google/vit-base-patch16-224-in21k",
                cache_dir=model_path
            )

        # 冻结参数
        if freeze_params:
            self._freeze_parameters()

        # 获取hidden_size用于后续任务
        self.hidden_size = self.vit.config.hidden_size

    def _freeze_parameters(self):
        """冻结所有ViT参数"""
        for param in self.vit.parameters():
            param.requires_grad = False
        print("ViT参数已冻结")

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像, shape: (batch_size, channels, height, width)

        Returns:
            dict: 包含cls_token和img_token的字典
        """
        # ViT forward pass
        outputs = self.vit(pixel_values=x)

        # 获取cls_token (第一个token)
        # shape: (batch_size, 1, hidden_size)
        cls_token = outputs.last_hidden_state[:, 0:1, :]

        # 获取img_token (所有patch tokens)
        # shape: (batch_size, num_patches, hidden_size)
        img_token = outputs.last_hidden_state[:, 1:, :]

        return {
            'cls_token': cls_token,  # [B, 1, hidden_size]
            'img_token': img_token,  # [B, num_patches, hidden_size]
            'last_hidden_state': outputs.last_hidden_state,  # 完整输出
            'pooler_output': outputs.pooler_output  # 分类头输出
        }

    def get_output_dimensions(self):
        """获取输出维度信息"""
        config = self.vit.config
        num_patches = (config.image_size // config.patch_size) ** 2
        return {
            'hidden_size': config.hidden_size,
            'num_patches': num_patches,
            'num_heads': config.num_attention_heads,
            'num_layers': config.num_hidden_layers
        }