import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import csv
import argparse
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F

from dataset import ProteinGODataset

from model_ph2_bp_adptivea import DeepAMIGO 

from metrics import compute_cafa_metrics
from goTerm_process import load_group_indices

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=2, gamma_pos=0, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y)
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        loss = (los_pos + los_neg) * one_sided_w
        return -loss.mean()


def save_log(path, epoch, train_loss, val_results, test_results, domain):
    file_exists = os.path.isfile(path)
    headers = ["Epoch", "Time", "Train_Loss"]
    row = [epoch, datetime.now().strftime("%H:%M:%S"), f"{train_loss:.4f}"]
    

    if domain in val_results:
        headers.extend([f"Val_{domain}_Fmax", f"Val_{domain}_AUPR", f"Val_{domain}_AUC"])
        row.extend([f"{val_results[domain]['Fmax']:.4f}", f"{val_results[domain]['AUPR_Macro']:.4f}", f"{val_results[domain]['AUC']:.4f}"])
            
    if domain in test_results:
        headers.extend([f"Test_{domain}_Fmax", f"Test_{domain}_AUPR", f"Test_{domain}_AUC"])
        row.extend([f"{test_results[domain]['Fmax']:.4f}", f"{test_results[domain]['AUPR_Macro']:.4f}", f"{test_results[domain]['AUC']:.4f}"])
            
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(headers)
        writer.writerow(row)

def run_epoch(model, loader, criterion, optimizer, indices_dict, device, target_domain, mode='train'):
    if mode == 'train':
        model.train()
        grad_ctx = torch.enable_grad
    else:
        model.eval()
        grad_ctx = torch.no_grad

    total_loss = 0
    all_logits = []
    all_targets = []
    
    domain_indices = indices_dict[target_domain]
    pbar = tqdm(loader, desc=f"{mode.upper()} [{target_domain}]", leave=False)
    
    with grad_ctx():
        for batch in pbar:
            batch = batch.to(device)
            targets = batch.y.float()
            logits = model(batch)
            if targets.shape != logits.shape: targets = targets.view(logits.shape)
            
            domain_logits = logits[:, domain_indices]
            domain_targets = targets[:, domain_indices]
            
            loss = criterion(domain_logits, domain_targets)
            total_loss += loss.item()

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            else:
                all_logits.append(domain_logits.detach().cpu())
                all_targets.append(domain_targets.detach().cpu())
    
    avg_loss = total_loss / len(loader)
    results = {}

    if mode != 'train':
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_probs = torch.sigmoid(all_logits)
        
        metrics = compute_cafa_metrics(all_targets, all_probs)
        results[target_domain] = metrics
            
    return avg_loss, results

def predict_epoch(model, loader, device, domain_indices):
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="ENSEMBLE PREDICT", leave=False):
            batch = batch.to(device)
            logits = model(batch)
            targets = batch.y.float()
            if targets.shape != logits.shape: targets = targets.view(logits.shape)
            
            all_logits.append(logits[:, domain_indices].cpu())
            all_targets.append(targets[:, domain_indices].cpu())
            
    all_probs = torch.sigmoid(torch.cat(all_logits, dim=0))
    all_targets = torch.cat(all_targets, dim=0)
    return all_probs, all_targets

def main(args):
    save_dir = f"./checkpoints_apa_{args.domain}"
    log_path = f"./training_log_apa_{args.domain}.csv"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f" Device: {args.device} |  Target Domain: {args.domain}")
    print(f" Loading Indices from {args.indices_path}...")
    indices_dict = load_group_indices(args.indices_path)
    if args.domain not in indices_dict:
        raise ValueError(f"Domain {args.domain} not found in indices_dict!")

    print(" Loading Datasets...")
    train_ds = ProteinGODataset(args.data_root, split='train')
    val_ds = ProteinGODataset(args.data_root, split='val')
    test_ds = ProteinGODataset(args.data_root, split='test')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = DeepAMIGO(num_classes=args.num_classes).to(args.device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias', 'scale_base', 'spline_weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=args.patience, verbose=True)
    
    criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=0, clip=0.05).to(args.device)

    best_score = 0.0
    early_stop = 0
    top_k_records = [] 
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} [{args.domain} Specialization]")
        
        train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, indices_dict, args.device, args.domain, 'train')
        val_loss, val_res = run_epoch(model, val_loader, criterion, None, indices_dict, args.device, args.domain, 'eval')
        test_loss, test_res = run_epoch(model, test_loader, criterion, None, indices_dict, args.device, args.domain, 'test')
        
        val_fmax = val_res[args.domain]['Fmax']
        val_aupr = val_res[args.domain]['AUPR_Macro']
        val_auc = val_res[args.domain]['AUC']
        test_fmax = test_res[args.domain]['Fmax']
        test_aupr = test_res[args.domain]['AUPR_Macro']
        test_auc = test_res[args.domain]['AUC']
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val  [{args.domain}]: Fmax={val_fmax:.4f} AUPR={val_aupr:.4f} AUC={val_auc:.4f}")
        print(f"Test [{args.domain}]: Fmax={test_fmax:.4f} AUPR={test_aupr:.4f} AUC={test_auc:.4f}")
        
        save_log(log_path, epoch+1, train_loss, val_res, test_res, args.domain)
        
        current_score = val_fmax 
        scheduler.step(current_score)
        
        ckpt_path = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), ckpt_path)
        top_k_records.append((current_score, ckpt_path))
        
        top_k_records.sort(key=lambda x: x[0], reverse=True)
        if len(top_k_records) > args.top_k_ensemble:
            _, removed_path = top_k_records.pop(-1)
            if os.path.exists(removed_path): os.remove(removed_path)
                
        if current_score > best_score:
            best_score = current_score
            early_stop = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f" {args.domain} new (Fmax: {best_score:.4f})")
        else:
            early_stop += 1
            print(f"Patience: {early_stop}/{args.patience}")
            
        if early_stop >= args.patience:
            print(f"early stop{args.domain} ")
            break


    print(f" {args.domain} Top-{args.top_k_ensemble} ")

    
    ensemble_probs = []
    final_targets = None
    domain_indices = indices_dict[args.domain]
    
    for rank, (score, path) in enumerate(top_k_records):
        print(f" Rank {rank+1}  (Val {args.domain} Fmax: {score:.4f}) ...")
        model.load_state_dict(torch.load(path))
        probs, targets = predict_epoch(model, test_loader, args.device, domain_indices)
        ensemble_probs.append(probs)
        if final_targets is None: final_targets = targets
            
   
    avg_probs = torch.stack(ensemble_probs).mean(dim=0)
    
    print(f" {args.domain} score：")
    metrics = compute_cafa_metrics(final_targets, avg_probs)
    print(f" [{args.domain}] Fmax: {metrics['Fmax']:.4f}  |  AUPR: {metrics['AUPR_Macro']:.4f}  |  AUC: {metrics['AUC']:.4f}") 
    
    ensemble_row = [
        f"Ensemble_Top{args.top_k_ensemble}",  
        datetime.now().strftime("%H:%M:%S"),   
        "-",                                   
        "-",                                   
        "-",                                   
        "-",                               
        f"{metrics['Fmax']:.4f}",              
        f"{metrics['AUPR_Macro']:.4f}",         
        f"{metrics['AUC']:.4f}" 
    ]
    
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ensemble_row)
    print(f"log {log_path} ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, required=True, choices=['MF', 'BP', 'CC'], help="Target domain to train")
    parser.add_argument('--data_root', type=str, default='../human_data')
    parser.add_argument('--indices_path', type=str, default='../HUMAN_target_indices.json')
    parser.add_argument('--num_classes', type=int, default=1244)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--top_k_ensemble', type=int, default=3)
    
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    main(args)