import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    """
    ViT 是 encoder-only 的结构
    dim = 768, 特征维度
    """
    def __init__(self, image_size = 224, patch_size = 16, num_classes = 1000,
                 dim = 768, depth = 12, heads = 12, mlp_dim = 3872):
        
        super(VisionTransformer, self).__init__()

        self.image_size = image_size
        # 切分的块大小
        self.patch_size = patch_size

        # 计算总的图像数量
        self.num_patches = (image_size // patch_size) ** 2

        # 每个图像块的维度
        self.patch_dim = 3 * patch_size ** 2

        self.conv = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

        # 特殊的 cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim), num_layers=depth)\
        
        self.layer_norm = nn.LayerNorm(dim)

        self.fc = nn.Linear(dim, num_classes)


    def forward(self, x):
        # 输入 x 的维度是 [batch_size, channels, height, width]，如：[batch_size, 3, 224, 224]

        # [batch_size, dim, height/patch_size, width/patch_size]=[batch_size, dim, 14, 14]
        x = self.conv(x)
        print(x.shape)
        
        # 然后 flatten 所有的空间维度height 和 width，将每个 patch 展开为一个向量
        # batch_size, num_patches, patch_dim] = [batch_size, 196, 768]
        x = x.flatten(2).transpose(1, 2)
        print(x.shape)
        
        # 生成并扩展 cls_token，使其与图像的 patch 数量匹配
        # [batch_size, 1, dim]，扩展为 [batch_size, 1, 768]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        print(x.shape)
        
        
        # 将 cls_token 和图像的 patch 拼接在一起，形成新的输入序列
        # [batch_size, 197, 768]， 197 是 196 个 patches + 1 个 cls token
        x = torch.cat((cls_tokens, x), dim=1)
        print(x.shape)

        # 通过 transformer 编码器进行处理
        # [batch_size, 197, 768]
        x = self.transformer_encoder(x)
        print(x.shape)
        
        # 选择第一个 token（cls_token）作为分类的输入
        # batch_size, 768]，只保留 cls_token 对应的输出
        x = x[:, 0]
        print(x.shape)

        # 对 cls_token 的输出应用 LayerNorm 进行归一化
        # [batch_size, 768]
        x = self.layer_norm(x)
        print(x.shape)
        
        # [batch_size, num_classes]
        return self.fc(x)
    
if __name__ == "__main__":
    model = VisionTransformer()
    print(model)
    x = torch.randn(1, 3, 224, 224)
    print(model(x))



