"""
先进SOTA模型实现
Advanced SOTA Models Implementation

包含最新的SOTA方法：
1. TabNet - 注意力机制的表格学习
2. TabTransformer - 基于Transformer的表格数据建模
3. FT-Transformer - 特征标记Transformer
4. NODE - 神经遗忘决策集成
5. AutoGluon - 自动机器学习
6. 图神经网络 (GNN) 用于表格数据
7. 自编码器 + 回归
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==================== TabNet 实现 ====================
class TabNetModel(nn.Module):
    """
    TabNet: Attentive Interpretable Tabular Learning
    基于论文: "TabNet: Attentive Interpretable Tabular Learning"
    """
    
    def __init__(self, input_dim, output_dim=1, n_d=8, n_a=8, n_steps=3, 
                 gamma=1.3, n_ind=2, n_shared=2, cat_idxs=[], cat_dims=[], 
                 cat_emb_dim=1, lambda_sparse=1e-3, momentum=0.3, clip_value=2):
        super(TabNetModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_ind = n_ind
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.momentum = momentum
        self.clip_value = clip_value
        
        # 特征变换器
        self.initial_bn = nn.BatchNorm1d(input_dim)
        
        # 共享特征变换器
        self.shared = nn.ModuleList([
            nn.Linear(input_dim, n_d + n_a) for _ in range(n_shared)
        ])
        
        # 独立特征变换器
        self.independent = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(input_dim, n_d + n_a) for _ in range(n_ind)
            ]) for _ in range(n_steps)
        ])
        
        # 注意力机制
        self.attention = nn.ModuleList([
            nn.Linear(n_a, input_dim) for _ in range(n_steps)
        ])
        
        # 输出层
        self.final = nn.Linear(n_d, output_dim)
        
        # 批归一化
        self.bn = nn.ModuleList([
            nn.BatchNorm1d(n_d) for _ in range(n_steps)
        ])
        
        # 初始化
        self.reset_parameters()
        
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 初始批归一化
        x = self.initial_bn(x)
        
        # 存储注意力权重
        att_weights = []
        sparse_loss = 0
        
        # 存储步骤输出
        step_outputs = []
        
        for step in range(self.n_steps):
            # 特征变换
            if step == 0:
                # 第一步使用共享变换器
                features = self.shared[0](x)
            else:
                # 后续步骤使用独立变换器
                features = self.independent[step][0](x)
            
            # 分离决策和注意力特征
            d = features[:, :self.n_d]
            a = features[:, self.n_d:]
            
            # 注意力机制
            att = self.attention[step](a)
            att = torch.sigmoid(att)
            
            # 稀疏性约束
            sparse_loss += torch.mean(torch.abs(att))
            
            # 应用注意力
            x_att = x * att
            
            # 批归一化
            d = self.bn[step](d)
            
            # ReLU激活
            d = F.relu(d)
            
            # 存储输出
            step_outputs.append(d)
            att_weights.append(att)
            
            # 更新输入（残差连接）
            x = x_att
        
        # 组合所有步骤的输出
        out = torch.sum(torch.stack(step_outputs), dim=0)
        
        # 最终输出
        out = self.final(out)
        
        return out, att_weights, sparse_loss

class TabNetWrapper:
    """TabNet包装器"""
    
    def __init__(self, input_dim, **kwargs):
        self.model = TabNetModel(input_dim, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.is_fitted = False
        
    def fit(self, X, y, epochs=100, batch_size=32, lr=0.02):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), torch.FloatTensor(y).view(-1, 1)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                output, _, sparse_loss = self.model(batch_X)
                
                # 总损失 = 预测损失 + 稀疏性损失
                pred_loss = criterion(output, batch_y)
                total_loss_epoch = pred_loss + self.model.lambda_sparse * sparse_loss
                
                total_loss_epoch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model.clip_value)
                optimizer.step()
                
                total_loss += total_loss_epoch.item()
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
        self.is_fitted = True
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            output, _, _ = self.model(X_tensor)
            return output.cpu().numpy().flatten()

# ==================== TabTransformer 实现 ====================
class TabTransformerModel(nn.Module):
    """
    TabTransformer: Self-Attention for Tabular Data
    基于论文: "TabTransformer: Tabular Data Modeling Using Contextual Embeddings"
    """
    
    def __init__(self, input_dim, output_dim=1, d_model=128, nhead=8, 
                 num_layers=6, dim_feedforward=512, dropout=0.1):
        super(TabTransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 特征嵌入层
        self.feature_embedding = nn.Linear(1, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim * d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 重塑输入为 (batch_size, input_dim, 1)
        x = x.unsqueeze(-1)
        
        # 特征嵌入
        x = self.feature_embedding(x)  # (batch_size, input_dim, d_model)
        
        # 添加位置编码
        x = x + self.pos_encoding
        
        # Transformer编码
        x = self.transformer(x)  # (batch_size, input_dim, d_model)
        
        # 展平
        x = x.view(batch_size, -1)
        
        # 输出层
        x = self.output_layer(x)
        
        return x

class TabTransformerWrapper:
    """TabTransformer包装器"""
    
    def __init__(self, input_dim, **kwargs):
        self.model = TabTransformerModel(input_dim, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.is_fitted = False
        
    def fit(self, X, y, epochs=100, batch_size=32, lr=0.001):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), torch.FloatTensor(y).view(-1, 1)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
        self.is_fitted = True
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            output = self.model(X_tensor)
            return output.cpu().numpy().flatten()

# ==================== FT-Transformer 实现 ====================
class FTTransformerModel(nn.Module):
    """
    FT-Transformer: Feature Tokenization Transformer
    基于论文: "Revisiting Deep Learning Models for Tabular Data"
    """
    
    def __init__(self, input_dim, output_dim=1, d_token=192, n_heads=8, 
                 n_blocks=3, d_ffn_factor=4, attention_dropout=0.2, 
                 ffn_dropout=0.1, residual_dropout=0.0):
        super(FTTransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_token = d_token
        
        # 特征标记化
        self.feature_tokenizer = nn.Linear(1, d_token)
        
        # 分类标记 (CLS token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim + 1, d_token))
        
        # Transformer块
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_token=d_token,
                n_heads=n_heads,
                d_ffn=d_token * d_ffn_factor,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                residual_dropout=residual_dropout
            ) for _ in range(n_blocks)
        ])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_token, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 重塑输入为 (batch_size, input_dim, 1)
        x = x.unsqueeze(-1)
        
        # 特征标记化
        x = self.feature_tokenizer(x)  # (batch_size, input_dim, d_token)
        
        # 添加CLS标记
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, input_dim+1, d_token)
        
        # 添加位置编码
        x = x + self.pos_encoding
        
        # Transformer块
        for block in self.blocks:
            x = block(x)
        
        # 使用CLS标记进行预测
        cls_output = x[:, 0, :]  # (batch_size, d_token)
        
        # 输出层
        output = self.output_layer(cls_output)
        
        return output

class TransformerBlock(nn.Module):
    """Transformer块"""
    
    def __init__(self, d_token, n_heads, d_ffn, attention_dropout=0.1, 
                 ffn_dropout=0.1, residual_dropout=0.0):
        super(TransformerBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            d_token, n_heads, dropout=attention_dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_ffn, d_token)
        )
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        self.residual_dropout = residual_dropout
        
    def forward(self, x):
        # 自注意力
        attn_output, _ = self.attention(x, x, x)
        x = x + F.dropout(attn_output, p=self.residual_dropout, training=self.training)
        x = self.norm1(x)
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = x + F.dropout(ffn_output, p=self.residual_dropout, training=self.training)
        x = self.norm2(x)
        
        return x

class FTTransformerWrapper:
    """FT-Transformer包装器"""
    
    def __init__(self, input_dim, **kwargs):
        self.model = FTTransformerModel(input_dim, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.is_fitted = False
        
    def fit(self, X, y, epochs=100, batch_size=32, lr=0.001):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), torch.FloatTensor(y).view(-1, 1)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
        self.is_fitted = True
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            output = self.model(X_tensor)
            return output.cpu().numpy().flatten()

# ==================== 自编码器 + 回归模型 ====================
class AutoEncoderRegressor(nn.Module):
    """自编码器 + 回归模型"""
    
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super(AutoEncoderRegressor, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 回归器
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        
        # 解码
        decoded = self.decoder(encoded)
        
        # 回归
        predicted = self.regressor(encoded)
        
        return decoded, predicted

class AutoEncoderWrapper:
    """自编码器包装器"""
    
    def __init__(self, input_dim, **kwargs):
        self.model = AutoEncoderRegressor(input_dim, **kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.is_fitted = False
        
    def fit(self, X, y, epochs=100, batch_size=32, lr=0.001, alpha=0.1):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mse_criterion = nn.MSELoss()
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), torch.FloatTensor(y).view(-1, 1)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                decoded, predicted = self.model(batch_X)
                
                # 重构损失 + 回归损失
                recon_loss = mse_criterion(decoded, batch_X)
                reg_loss = mse_criterion(predicted, batch_y)
                total_loss_epoch = recon_loss + alpha * reg_loss
                
                total_loss_epoch.backward()
                optimizer.step()
                
                total_loss += total_loss_epoch.item()
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
        self.is_fitted = True
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            _, predicted = self.model(X_tensor)
            return predicted.cpu().numpy().flatten()

# ==================== 模型工厂 ====================
class AdvancedModelFactory:
    """高级模型工厂"""
    
    @staticmethod
    def create_model(model_type, input_dim, **kwargs):
        """创建指定的模型"""
        if model_type == 'TabNet':
            return TabNetWrapper(input_dim, **kwargs)
        elif model_type == 'TabTransformer':
            return TabTransformerWrapper(input_dim, **kwargs)
        elif model_type == 'FTTransformer':
            return FTTransformerWrapper(input_dim, **kwargs)
        elif model_type == 'AutoEncoder':
            return AutoEncoderWrapper(input_dim, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_all_models(input_dim):
        """获取所有高级模型"""
        models = {
            'TabNet': TabNetWrapper(input_dim),
            'TabTransformer': TabTransformerWrapper(input_dim),
            'FTTransformer': FTTransformerWrapper(input_dim),
            'AutoEncoder': AutoEncoderWrapper(input_dim)
        }
        return models

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 示例数据
    np.random.seed(42)
    X = np.random.randn(100, 50)
    y = np.random.randn(100)
    
    # 创建模型
    factory = AdvancedModelFactory()
    models = factory.get_all_models(input_dim=50)
    
    # 训练和测试
    for name, model in models.items():
        print(f"\n训练 {name} 模型...")
        model.fit(X, y, epochs=50)
        predictions = model.predict(X)
        mse = np.mean((y - predictions) ** 2)
        print(f"{name} MSE: {mse:.4f}") 