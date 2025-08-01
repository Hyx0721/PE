# Imports
import numpy as np
import torch
import torch.nn as nn
from utils import ReverseLayerF
import torch.nn.functional as F
import math
import torchvision
#import models
import os.path as osp
    
class ShallowConvNet(nn.Module):
    def __init__(self, in_channels=1, out_channels1=8, out_channels2=8, n_classes=6, drop_out=0.5):
        super(ShallowConvNet, self).__init__()

        self.drop_out = drop_out
        self.weights = nn.Parameter(torch.Tensor(1, 60, 500))  

        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1, kernel_size=(40, 60), padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_channels1), 
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7)), 
            nn.Dropout(drop_out)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(out_channels1, out_channels2, kernel_size=(16, 8), padding=(0, 2), bias=False),
            nn.BatchNorm2d(out_channels2), 
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 13), stride=(1, 4)), 
            nn.Dropout(drop_out)
        )


    def forward(self, x):
        x = x * self.weights  
        x = self.block_1(x)
        x = self.block_2(x)

        x = x.view(x.size(0), -1)
        #print(x.shape)

        return x
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.drop_out = 0.25
        self.C = 60
        #left-right:26
        #frontal=23
        self.T = 500
        #500
        self.weights = nn.Parameter(torch.Tensor(1, self.C, self.T))
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((30, 31, 0, 0)),
            nn.Conv2d(
                in_channels=1,          
                out_channels=8,         
                kernel_size=(1, self.C),    
                bias=False
            ),                        
            nn.BatchNorm2d(8)           
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,          
                out_channels=16,        
                kernel_size=(self.C, 1), 
                groups=8,
                bias=False
            ),                        
            nn.BatchNorm2d(16),       
            nn.ELU(),
            nn.AvgPool2d((1, 4)),    
            nn.Dropout(self.drop_out)
        )
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
               in_channels=16,      
               out_channels=16,       
               kernel_size=(1, 16),  
               groups=16,
               bias=False
            ),                       
            nn.Conv2d(
                in_channels=16,        
                out_channels=32,    
                kernel_size=(1, 1), 
                bias=False
            ),                    
            nn.BatchNorm2d(32),         
            nn.ELU(),
            nn.AvgPool2d((1, 8)),    
            nn.Dropout(self.drop_out)
        )

    def reset_parameters(self):
        stdv = 1. / math.sqrt(1)
        self.weights.data.uniform_(-stdv, stdv)
    def forward(self, x):
        # x = x*self.weights
        x = self.block_1(x)
        x = self.block_2(x)
        # x = self.block_3(x)
        # x = x.view(x.size(0), -1)
        feature_map = self.block_3(x)  
        x = feature_map.view(feature_map.size(0), -1)  # 展平输出
        # x = self.linear(x)
        
        return x
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.04)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

from torchvision import  models      
class BMCL(nn.Module):
    def __init__(self, config):
        super(BMCL, self).__init__()

        self.config = config
        self.EEG_size = config.embedding_size           #embedding_size=300
        self.image_size = config.embedding_size

        self.input_sizes = input_sizes = [self.EEG_size, self.image_size]
        self.hidden_sizes = hidden_sizes = [int(self.EEG_size), int(self.image_size)]           #hidden_sizes=128
        self.output_size = output_size = 7              #class_opt = 7
        self.dropout_rate = dropout_rate = config.dropout                  #0.5
        self.activation = self.config.activation()                                      #relu
        self.tanh = nn.Tanh()
        
        if self.config.data == 'facial':
            #Enc_eeg()
            self.eeg_model = EEGNet()              # EEGNet
            feature_extractor = models.resnet34(pretrained=True)
            self.image_model = Image(feature_extractor, droprate=0.5, pretrained=True)

            self.eeg_size = 480
            #left brain-right=512
            #all=480
            #16*15
            self.image_size =4096

        for param in self.eeg_model.parameters():
            param.requires_grad = self.config.requires_grad
        # for param in self.image_model.parameters():
        #     param.requires_grad = self.config.requires_grad
        
        self.fusion_xyz = nn.Sequential()
        self.fusion_xyz.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size,
                                                            out_features=self.config.hidden_size))
        self.fusion_xyz.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion_xyz.add_module('fusion_layer_1_activation', self.activation)
        self.fusion_xyz.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size,
                                                            out_features=6))
        
        self.fusion_e_xyz = nn.Sequential()
        self.fusion_e_xyz.add_module('fusion_layer_1', nn.Linear(in_features=128,
                                                                 #self.eeg_size,
                                                            out_features=self.config.hidden_size))
        self.fusion_e_xyz.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion_e_xyz.add_module('fusion_layer_1_activation', self.activation)
        self.fusion_e_xyz.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size,
                                                            out_features=6))
        self.fusion_i_xyz = nn.Sequential()
        self.fusion_i_xyz.add_module('fusion_layer_1', nn.Linear(in_features=4096,
                                                                 #self.image_size,
                                                            out_features=self.config.hidden_size))
        self.fusion_i_xyz.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion_i_xyz.add_module('fusion_layer_1_activation', self.activation)
        self.fusion_i_xyz.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size,
                                                            out_features=6))
        
        self.eeg_transformer = CustomTransformerLayer(60, num_heads=1)  
        self.image_predictor = nn.Sequential(
            nn.Linear(self.eeg_size, 128),
            nn.ReLU(),
            nn.Linear(128, self.image_size)  
        )


        self.cross2=BiCrossAttention_EEGOnly(hidden_dim=128)

    def alignment(self, eeg, image):

  
  
  
        representation_eeg = self.eeg_model(eeg)
        #print(representation_eeg.shape)
        B = representation_eeg.size(0) 
        FE = representation_eeg.view(B, 8, 60)         
        #FE = representation_eeg.view(B, 16, 33)            
        FE = FE.permute(1, 0, 2)                             
        representation_eeg = self.eeg_transformer(FE, FE, FE).reshape(B, -1)
        
        representation_image = self.image_model(image)

        h_e=self.cross2(representation_eeg,representation_image)

        h_i=representation_image
        o_e=self.fusion_e_xyz(h_e)
        o_i=self.fusion_i_xyz(h_i)
 
 
        return o_e,o_i

        

    def forward(self, eeg, image):
        self.eeg = eeg
        self.image = image

        o_e,o_i = self.alignment(eeg, image)
        
        return o_e,o_i

class Image(nn.Module):
    def __init__(self, feature_extractor, droprate=0.5, pretrained=True, feat_dim=4096):
        super(Image, self).__init__()
        self.droprate = droprate
        feature_extractor = models.resnet34(pretrained=True)
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)                #修改平均池化层其输出大小为(1, 1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)             #新全连接层的输入特征数设置为fe_out_planes（即原始全连接层的输入特征数），
        self.att = AttentionBlock(feat_dim)               #注意力块（AttentionBlock）


        if pretrained:
            init_modules = [self.feature_extractor.fc]
        else:
            init_modules = self.modules()
        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):          #检查它是否是 nn.Conv2d（二维卷积层）或 nn.Linear（全连接层）的实例
                nn.init.kaiming_normal_(m.weight.data)                    #则使用 Kaiming 初始化（也称为 He 初始化）来初始化权重，并将偏置（如果有的话）初始化为 0。
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    def forward(self, x):
        x = self.feature_extractor(x)                  #特征提取器（self.feature_extractor）处理输入 x，然后应用 ReLU 激活函数。
        x = F.relu(x)
        x = self.att(x.view(x.size(0), -1))
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate) 
        return x            #返回特征
   
class CustomTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CustomTransformerLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, Q, K, V):
        # Q, K, V shape: (seq_length, batch_size, embed_dim)
        # Q = Q.permute(2, 0, 1)
        # K = K.permute(2, 0, 1)
        # V = V.permute(2, 0, 1)
        attn_output, attn_output_weights = self.multihead_attn(Q, K, V)
        Q = self.norm1(Q + attn_output)
        ff_output = F.relu(self.linear1(Q))
        ff_output = self.dropout(ff_output)
        ff_output = self.linear2(ff_output)
        output = self.norm2(Q + ff_output)

        return output.permute(1, 2, 0)
    


import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, input_dim_q, input_dim_kv, hidden_dim=128, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.scale = (hidden_dim // heads) ** -0.5
        
        # Project to common hidden dimension
        self.proj_q = nn.Linear(input_dim_q, hidden_dim)
        self.proj_k = nn.Linear(input_dim_kv, hidden_dim)
        self.proj_v = nn.Linear(input_dim_kv, hidden_dim)
        
        # Multi-head attention output
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Residual projection (if dimensions don't match)
        self.residual_proj = None
        if input_dim_q != hidden_dim:
            self.residual_proj = nn.Linear(input_dim_q, hidden_dim)
    
    def forward(self, q_input, kv_input):
        # Assume input shape: (B, C) —> unsqueeze to (B, 1, C)
        B = q_input.size(0)
        q_input_orig = q_input.unsqueeze(1)  # Keep original for residual
        q_input = q_input.unsqueeze(1)  # (B, 1, 480)
        kv_input = kv_input.unsqueeze(1)  # (B, 1, 4096)
        
        # Project
        q = self.proj_q(q_input)  # (B, 1, hidden_dim)
        k = self.proj_k(kv_input)
        v = self.proj_v(kv_input)
        
        # Split heads
        q = q.view(B, 1, self.heads, self.hidden_dim // self.heads).transpose(1, 2)  # (B, heads, 1, dim_head)
        k = k.view(B, 1, self.heads, self.hidden_dim // self.heads).transpose(1, 2)
        v = v.view(B, 1, self.heads, self.hidden_dim // self.heads).transpose(1, 2)
        
        # Attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, 1, 1)
        attn = attn_scores.softmax(dim=-1)
        out = attn @ v  # (B, heads, 1, dim_head)
        
        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, 1, self.hidden_dim)
        out = self.to_out(out)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(q_input_orig)
        else:
            residual = self.proj_q(q_input_orig)  # Use projected query as residual
        
        out = out + residual  # Add residual connection
        
        return out.squeeze(1)  # (B, hidden_dim)

class BiCrossAttention_EEGOnly(nn.Module):
    def __init__(self, dim_eeg=544, dim_img=4096, hidden_dim=128):
        super().__init__()
        # Only EEG as query, Image as key-value
        self.eeg_from_img = CrossAttention(input_dim_q=dim_eeg, input_dim_kv=dim_img, hidden_dim=hidden_dim)
        
        # Residual projection for EEG output only
        self.eeg_residual_proj = None
        if dim_eeg != hidden_dim:
            self.eeg_residual_proj = nn.Linear(dim_eeg, hidden_dim)
    
    def forward(self, eeg_feat, img_feat):
        # Cross attention: EEG ← Image only
        fused_eeg = self.eeg_from_img(q_input=eeg_feat, kv_input=img_feat)  # (B, hidden_dim)
        
        # Residual connection for EEG
        if self.eeg_residual_proj is not None:
            eeg_residual = self.eeg_residual_proj(eeg_feat)
        else:
            eeg_residual = eeg_feat
            
        fused_eeg = fused_eeg + eeg_residual  # (B, hidden_dim)
        
        return fused_eeg  # Only return fused EEG features

class AttentionBlock(nn.Module):
    
    def __init__(self, in_channels):               #inchannel ==2048
        super(AttentionBlock, self).__init__()
        self.g = nn.Linear(in_channels, in_channels // 8)
        self.theta = nn.Linear(in_channels, in_channels // 8)
        self.phi = nn.Linear(in_channels, in_channels // 8)

        self.W = nn.Linear(in_channels // 8, in_channels)

    def forward(self, x):
        batch_size = x.size(0)
        out_channels = x.size(1)

        g_x = self.g(x).view(batch_size, out_channels // 8, 1)                #被重新塑形（reshape）为一个三维张量，其维度分别为 batch_size、out_channels // 8 和 1。

        theta_x = self.theta(x).view(batch_size, out_channels // 8, 1)
        theta_x = theta_x.permute(0, 2, 1)                                          #将 theta_x 的维度重新排列为 (batch_size, 1, out_channels // 8)
        phi_x = self.phi(x).view(batch_size, out_channels // 8, 1)

        f = torch.matmul(phi_x, theta_x)                      #进行矩阵乘法
        f_div_C = F.softmax(f, dim=-1)                           #可以将一个向量压缩到另一个向量中，使得每一个元素的范围都在 (0, 1) 之间，并且所有元素的和为 1

        y = torch.matmul(f_div_C, g_x)

        y = y.view(batch_size, out_channels // 8)
        W_y = self.W(y)
        z = W_y + x
        return z
   