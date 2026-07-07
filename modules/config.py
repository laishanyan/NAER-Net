import torch
# from pytorch_pretrained import BertTokenizer
from transformers import BertModel, BertTokenizer

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'Fuzzy-MSF'                                                   # 模型保存名称
        self.train_path = dataset + '/dataset/train.txt'                                # 训练集
        self.dev_path = dataset + '/dataset/test.txt'                                    # 验证集
        self.test_path = dataset + '/dataset/test.txt'                                  # 测试集
        self.dir_path = dataset + '/dataset/dir.txt'                                    # 灾害损失字典
        self.image_path = dataset + '/dataset/images/'
        self.class_list = [x.strip() for x in open(
            dataset + '/dataset/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/save_dict/' + self.model_name + '.ckpt'         # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 5000                                 # 若超过5000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 30                                            # epoch数
        self.batch_size = 24                                            # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.img_pad_size = 196                                         # 每张图片处理成的块数
        self.learning_rate = 1e-7                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.dropout = 0.2

        self.dim_model = 768
        self.embed = 768
        self.num_head = 4
        self.num_encoder = 1

        self.subspace_size = 256
        self.attn_hidden_size = 128

        self.vit_path = '/home/sylai/code/vit'