import torch
import os
import os.path as osp
from torch_geometric.data import Dataset

class ProteinGODataset(Dataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None):
        """
        Args:
            root (str):  ../../data/newdata)
            split (str): 'train', 'val', 'test'
        """
        self.split = split
        self.data_dir = osp.join(root, split)
        
        if not osp.exists(self.data_dir):
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")
            
        self.file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.pt')]
        super().__init__(root, transform, pre_transform)

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        filename = self.file_list[idx]
        data_path = osp.join(self.data_dir, filename)
        
        try:
            data = torch.load(data_path,weights_only=False)
        except Exception as e:
            print(f"Error loading {data_path}: {e}")
            raise e
        
        if hasattr(data, 'x_esm2'): data.x_esm2 = data.x_esm2.float()
        if hasattr(data, 'x_protT5'): data.x_protT5 = data.x_protT5.float()
        if hasattr(data, 'x_onehot'): data.x_onehot = data.x_onehot.float()
        
        if hasattr(data, 'x_esm2'):
            data.num_nodes = data.x_esm2.shape[0]
        elif hasattr(data, 'x_onehot'):
            data.num_nodes = data.x_onehot.shape[0]
        else:
            data.num_nodes = 0

        if hasattr(data, 'edge_index_3d') and hasattr(data, 'edge_attr_3d'):
            data.edge_index = data.edge_index_3d.long()
            data.edge_attr = data.edge_attr_3d.float()
        else:
            if hasattr(data, 'native_contact') and data.native_contact is not None:
                data.edge_index = data.native_contact.long()
            else:
                data.edge_index = torch.arange(data.num_nodes).unsqueeze(0).repeat(2, 1).long()
                
            original_num_edges = data.edge_index.size(1)
            data.edge_attr = torch.zeros((original_num_edges, 16), dtype=torch.float32)

        if data.edge_attr.dim() == 1:
            data.edge_attr = data.edge_attr.unsqueeze(-1)

        keys_to_remove = ['native_contact', 'pssm', 'seq', 'pid', 'raw_seq', 'edge_index_3d', 'edge_attr_3d']
        for key in keys_to_remove:
            if hasattr(data, key):
                delattr(data, key)

        return data