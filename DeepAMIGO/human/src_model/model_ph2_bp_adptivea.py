import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
from torch_geometric.nn import GlobalAttention
from torch_geometric.utils import softmax, to_dense_batch

class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels=21, out_channels=256, kernels=[3, 5, 7]):
        super(MultiScaleCNN, self).__init__()
        self.convs = nn.ModuleList()
        groups = min(32, out_channels)
        for k in kernels:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=(k-1)//2),
                    nn.GroupNorm(groups, out_channels), 
                    nn.ReLU()
                )
            )
        self.fusion = nn.Linear(out_channels * len(kernels), out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x_dense, mask):
        x = x_dense.transpose(1, 2) 
        outs = []
        for conv in self.convs:
            outs.append(conv(x)) 
        cat_out = torch.cat(outs, dim=1) 
        cat_out = cat_out.transpose(1, 2) 
        
        out = self.fusion(cat_out)
        
        mask_float = mask.unsqueeze(-1).float() 
        out = out * mask_float  
        
        out = self.norm(out)
        out = out * mask_float 
        out = self.dropout(out)
        
        return out[mask] 


class ResidualGINEBlock(nn.Module):
    def __init__(self, channels, edge_dim=1, dropout=0.5):
        super().__init__()
        nn_callable = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )
        self.conv = GINEConv(nn=nn_callable, edge_dim=edge_dim)
        self.norm = nn.LayerNorm(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        residual = x
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.norm(x + residual)
        return out


class DPFTransformerBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=4):
        super(DPFTransformerBlock, self).__init__()
        self.num_heads = num_heads
        
        self.trans_q_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(num_heads)])
        self.trans_k_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(num_heads)])
        self.trans_v_list = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(num_heads)])
        
        self.concat_trans = nn.Linear(hidden_dim * num_heads, hidden_dim, bias=False)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.layernorm1 = nn.LayerNorm(in_dim)
        self.layernorm2 = nn.LayerNorm(in_dim)
        
    def forward(self, x_residue, x_inter, batch_idx):
        inter_expanded = x_inter[batch_idx] 
        
        multi_output = []
        for i in range(self.num_heads):
            q = self.trans_q_list[i](inter_expanded)       
            k = self.trans_k_list[i](x_residue)  
            v = self.trans_v_list[i](x_residue)       
            
            att = torch.sum(q * k, dim=1, keepdim=True) / (x_residue.size(-1) ** 0.5)
            alpha = softmax(att, batch_idx) 
            
            tp = v * alpha
            multi_output.append(tp)
            
        multi_output = torch.cat(multi_output, dim=1)
        multi_output = self.concat_trans(multi_output)
        
        multi_output = self.layernorm1(multi_output + x_residue)
        multi_output = self.layernorm2(self.ff(multi_output) + multi_output)
        
        return multi_output


class DeepAMIGO(nn.Module): 
    def __init__(self, num_classes=2752, hidden_dim=256, cross_attention_heads=4, dropout=0.5):
        super(DeepAMIGO, self).__init__()
        
        
        self.alpha_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(), 
            nn.Linear(hidden_dim // 4, 1)
        )
        
        plm_in_dim = 2560 + 1024
        self.plm_proj = nn.Sequential(
            nn.Linear(plm_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.cnn_encoder = MultiScaleCNN(in_channels=21, out_channels=hidden_dim)
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.res_layers = nn.ModuleList([
            ResidualGINEBlock(hidden_dim, edge_dim=16, dropout=dropout),
            ResidualGINEBlock(hidden_dim, edge_dim=16, dropout=dropout)
        ])
        
        self.global_context_proc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
        self.attn_pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        )
        
        self.transformer = DPFTransformerBlock(hidden_dim, hidden_dim, num_heads=cross_attention_heads)
        
        final_dim = hidden_dim * 2 
            
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(final_dim),
            nn.Linear(final_dim, final_dim * 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(final_dim * 2, final_dim * 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(final_dim * 2, num_classes)
        )
    def get_graph_embedding(self, batch):
       
        with torch.no_grad():
            _ = self.forward(batch)          
            return self._last_embedding      

    def forward(self, data):
        if not hasattr(data, 'x_esm2') or not hasattr(data, 'x_protT5') or not hasattr(data, 'x_onehot'):
            raise ValueError("Model requires x_esm2, x_protT5, and x_onehot in data object")
        data.edge_index = data.edge_index.long()

        x_plm = torch.cat([data.x_esm2, data.x_protT5], dim=-1)
        h_plm = self.plm_proj(x_plm) 
        
        x_onehot_dense, mask = to_dense_batch(data.x_onehot, data.batch) 
        h_cnn = self.cnn_encoder(x_onehot_dense, mask) 
        
        cat_feat = torch.cat([h_plm, h_cnn], dim=-1)
        
        gate = self.fusion_gate(cat_feat)
        
        x_node_init = gate * h_plm + (1.0 - gate) * h_cnn
        
        init_feature = global_mean_pool(x_node_init, data.batch)
        
        global_context_raw = self.attn_pool(h_plm, data.batch)
        gate_scores = self.attn_pool.gate_nn(h_plm)              # [N, 1]
        self._node_attn_weights = softmax(gate_scores, data.batch).squeeze(-1).detach()
        
        global_context = self.global_context_proc(global_context_raw) 
        
        x_gcn = x_node_init
        edge_attr = getattr(data, 'edge_attr', None)
        
        if edge_attr is None:
            edge_attr = torch.ones((data.edge_index.size(1), 16), dtype=torch.float32, device=x_gcn.device)
        else:
            edge_attr = edge_attr.float()
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
        
        for layer in self.res_layers:
            x_gcn = layer(x_gcn, data.edge_index, edge_attr)
            
        x_attended = self.transformer(x_gcn, global_context, data.batch)
        
        

        adaptive_gate = torch.sigmoid(self.alpha_net(global_context))  # [B, 1]
        adaptive_gate_node = adaptive_gate[data.batch]  # [N, 1]
        x_structural_fused = adaptive_gate_node * x_node_init + (1 - adaptive_gate_node) * x_attended
        graph_feature = global_add_pool(x_structural_fused, data.batch)
        
        final_vec = torch.cat([init_feature, graph_feature], dim=1)
        self._last_embedding = final_vec
        logits = self.classifier(final_vec)
        
        return logits