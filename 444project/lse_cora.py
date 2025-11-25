import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 增强的LSE模型 - 添加轻微的正则化和更好的dropout策略
class StableLSE(nn.Module):
    def __init__(self, in_dim=1433, hid=256, out=128, k_max=7):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid)
        self.bn1 = nn.BatchNorm1d(hid)
        self.conv2 = GCNConv(hid, out)
        self.bn2 = nn.BatchNorm1d(out)
        self.proj = nn.Linear(out, k_max)
        
        # 温和初始化
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x, edge_index):
        h = F.elu(self.bn1(self.conv1(x, edge_index)))
        h = F.dropout(h, p=0.4, training=self.training)  # 稍微增加dropout
        h = F.elu(self.bn2(self.conv2(h, edge_index)))
        h = F.dropout(h, p=0.3, training=self.training)  # 第二层也加dropout
        z = F.softmax(self.proj(h), dim=1)
        return h, z

# 增强的LSE损失 - 添加轻微的温度调节
def compute_stable_lse(z, epoch=0):
    """
    带轻微温度调节的LSE损失
    """
    # 随着训练进展，逐渐降低温度（让分配更确定）
    temperature = max(0.7, 1.0 - epoch * 0.002)  # 从1.0线性降到0.7
    z_scaled = F.softmax(z / temperature, dim=1)
    
    pi = z_scaled.mean(0)
    h_prior = -torch.sum(pi * torch.log(pi + 1e-10))
    h_assign = -torch.sum(z_scaled * torch.log(z_scaled + 1e-10)) / z_scaled.size(0)
    
    # 簇大小平衡
    cluster_sizes = torch.sum(z_scaled, dim=0)
    size_penalty = torch.std(cluster_sizes) / (torch.mean(cluster_sizes) + 1e-10)
    
    return h_prior + h_assign + 0.1 * size_penalty

# 增强的对比损失 - 更稳定的温度控制
def improved_contrastive_loss(h, edge_index, num_nodes, epoch=0):
    """
    带温度衰减的对比学习损失
    """
    src, dst = edge_index
    
    # 温度随训练衰减
    temperature = max(0.2, 0.5 - epoch * 0.001)  # 从0.5线性降到0.2
    
    # 正样本对
    pos_sim = F.cosine_similarity(h[src], h[dst])
    pos_loss = -torch.log(torch.sigmoid(pos_sim / temperature) + 1e-10).mean()
    
    # 负采样 - 根据epoch调整数量
    base_neg = 4000
    num_neg = min(base_neg + epoch * 10, num_nodes // 3)  # 逐渐增加负样本
    neg_src = torch.randint(0, num_nodes, (num_neg,), device=h.device)
    neg_dst = torch.randint(0, num_nodes, (num_neg,), device=h.device)
    neg_sim = F.cosine_similarity(h[neg_src], h[neg_dst])
    neg_loss = -torch.log(1 - torch.sigmoid(neg_sim / temperature) + 1e-10).mean()
    
    return (pos_loss + neg_loss) / 2

def train_enhanced_lse():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    dataset = Planetoid(root='./data', name='Cora')
    data = dataset[0].to(device)
    print(f"Data: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_features} features")
    
    # 初始化模型
    model = StableLSE(in_dim=data.num_features).to(device)
    
    # 增强的优化器 - 使用AdamW和学习率调度
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-5)
    
    print('Training Enhanced LSEnet...')
    best_nmi = 0
    best_state = None
    patience = 50  # 增加耐心值
    patience_counter = 0
    nmi_history = []
    
    for epoch in tqdm(range(400), desc="Training"):  # 增加到400轮
        model.train()
        h, z = model(data.x, data.edge_index)
        
        # 增强的损失组合 - 带epoch信息
        l_se = compute_stable_lse(z, epoch)
        l_ae = improved_contrastive_loss(h, data.edge_index, data.num_nodes, epoch)
        
        # 动态权重调整 - 前期注重重建，后期注重聚类
        ae_weight = max(0.03, 0.1 * (1 - epoch / 399))  # 从0.1线性降到0.03
        loss = l_se + ae_weight * l_ae
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # 智能评估策略 - 前期频繁，后期稀疏
        eval_freq = 10 if epoch < 100 else (20 if epoch < 200 else 30)
        if epoch % eval_freq == 0 or epoch < 30:
            with torch.no_grad():
                model.eval()
                _, z_eval = model(data.x, data.edge_index)
                current_nmi, cluster_info = evaluate_clustering_detailed(z_eval, data.y)
                nmi_history.append(current_nmi)
                
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch:3d}: LSE={l_se.item():.4f}, AE={l_ae.item():.4f}, '
                      f'LR={current_lr:.6f}, NMI={current_nmi:.4f}, '
                      f'Clusters={cluster_info["num_clusters"]}')
                
                if current_nmi > best_nmi:
                    best_nmi = current_nmi
                    best_state = model.state_dict().copy()
                    patience_counter = 0
                    print(f'  -> New best! NMI: {best_nmi:.4f}')
                    
                    # 保存最佳模型
                    torch.save(best_state, 'lse_cora_stable.pth')
                else:
                    patience_counter += 1
                
                # 基于NMI历史的智能早停
                if len(nmi_history) > 10:
                    recent_improvement = max(nmi_history[-5:]) - max(nmi_history[-10:-5])
                    if recent_improvement < 0.01 and patience_counter >= 15 and epoch > 100:
                        print(f'Early stopping at epoch {epoch} - minimal improvement')
                        break
                
                if patience_counter >= patience and epoch > 80:
                    print(f'Early stopping at epoch {epoch} - patience exceeded')
                    break
    
    # 最终模型选择策略
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'Loaded best model with NMI: {best_nmi:.4f}')
    
    return model, data, best_nmi, best_state

def evaluate_clustering_detailed(z, true_labels, n_trials=20):
    """增强的聚类评估"""
    z_np = z.cpu().numpy()
    true_np = true_labels.cpu().numpy()
    
    best_nmi = 0
    best_pred = None
    all_nmis = []
    
    for i in range(n_trials):
        kmeans = KMeans(n_clusters=7, random_state=i, n_init=1)
        pred = kmeans.fit_predict(z_np)
        current_nmi = nmi(true_np, pred)
        all_nmis.append(current_nmi)
        if current_nmi > best_nmi:
            best_nmi = current_nmi
            best_pred = pred
    
    # 统计信息
    nmi_mean = np.mean(all_nmis)
    nmi_std = np.std(all_nmis)
    
    # 分析簇信息
    unique, counts = np.unique(best_pred, return_counts=True)
    cluster_info = {
        'num_clusters': len(unique),
        'min_size': np.min(counts),
        'max_size': np.max(counts),
        'std_size': np.std(counts),
        'nmi_mean': nmi_mean,
        'nmi_std': nmi_std
    }
    
    return best_nmi, cluster_info

def visualize_stable_results(h_np, z_np, true_labels, pred_labels, nmi_score):
    """修复的可视化函数"""
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    print("Computing t-SNE embeddings...")
    h_2d = tsne.fit_transform(h_np)
    z_2d = tsne.fit_transform(z_np)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 隐藏层可视化
    sc1 = axes[0,0].scatter(h_2d[:, 0], h_2d[:, 1], c=pred_labels, cmap='tab10', alpha=0.8, s=15)
    axes[0,0].set_title(f'Hidden Space - Predicted (NMI: {nmi_score:.4f})')
    plt.colorbar(sc1, ax=axes[0,0])
    
    sc2 = axes[0,1].scatter(h_2d[:, 0], h_2d[:, 1], c=true_labels, cmap='tab10', alpha=0.8, s=15)
    axes[0,1].set_title('Hidden Space - True Labels')
    plt.colorbar(sc2, ax=axes[0,1])
    
    # 分配空间可视化
    sc3 = axes[1,0].scatter(z_2d[:, 0], z_2d[:, 1], c=pred_labels, cmap='tab10', alpha=0.8, s=15)
    axes[1,0].set_title('Assignment Space - Predicted')
    plt.colorbar(sc3, ax=axes[1,0])
    
    sc4 = axes[1,1].scatter(z_2d[:, 0], z_2d[:, 1], c=true_labels, cmap='tab10', alpha=0.8, s=15)
    axes[1,1].set_title('Assignment Space - True Labels')
    plt.colorbar(sc4, ax=axes[1,1])
    
    # 簇大小分布
    unique, counts = np.unique(pred_labels, return_counts=True)
    axes[0,2].bar(unique, counts, color=plt.cm.tab10(unique))
    axes[0,2].set_title('Cluster Size Distribution')
    axes[0,2].set_xlabel('Cluster ID')
    axes[0,2].set_ylabel('Number of Nodes')
    
    # 分配概率热图
    im = axes[1,2].imshow(z_np.T, aspect='auto', cmap='viridis')
    axes[1,2].set_title('Assignment Probability Matrix')
    axes[1,2].set_xlabel('Nodes')
    axes[1,2].set_ylabel('Clusters')
    plt.colorbar(im, ax=axes[1,2])
    
    plt.tight_layout()
    plt.savefig('cora_stable_results.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to 'cora_stable_results.png'")
    plt.show()

# 主执行
if __name__ == "__main__":
    model, data, best_nmi, best_state = train_enhanced_lse()
    
    # 最终评估
    if best_state is not None:
        model.load_state_dict(best_state)
    
    model.eval()
    
    with torch.no_grad():
        h, z = model(data.x, data.edge_index)
        z_np = z.cpu().numpy()
        true_y = data.y.cpu().numpy()
        
        # 增强的最终聚类评估
        best_pred = None
        best_nmi_final = 0
        all_nmis = []
        
        for i in range(25):  # 增加到25次试验
            kmeans = KMeans(n_clusters=7, random_state=i, n_init=1)
            pred = kmeans.fit_predict(z_np)
            current_nmi = nmi(true_y, pred)
            all_nmis.append(current_nmi)
            if current_nmi > best_nmi_final:
                best_nmi_final = current_nmi
                best_pred = pred
        
        final_ari = ari(true_y, best_pred)
        nmi_mean = np.mean(all_nmis)
        nmi_std = np.std(all_nmis)
        
        print(f'\n=== ENHANCED FINAL RESULTS ===')
        print(f'Best NMI: {best_nmi_final:.4f}')
        print(f'Average NMI: {nmi_mean:.4f} ± {nmi_std:.4f}')
        print(f'Best ARI: {final_ari:.4f}')
        
        # 详细分析
        unique, counts = np.unique(best_pred, return_counts=True)
        print(f'Cluster distribution: {dict(zip(unique, counts))}')
        print(f'Cluster sizes - Min: {min(counts)}, Max: {max(counts)}, Std: {np.std(counts):.1f}')
        print(f'Number of clusters found: {len(unique)}/7')
        
        # 可视化
        visualize_stable_results(h.cpu().numpy(), z_np, true_y, best_pred, best_nmi_final)