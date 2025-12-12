import os
import copy
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class FocalCE(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalCE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class GatedAttentionThSig(nn.Module):
    def __init__(self, D=64, L=64, dropout=0.25):
        super(GatedAttentionThSig, self).__init__()
        self.tanhV = nn.Sequential(
            nn.Linear(D, L),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.sigmU = nn.Sequential(
            nn.Linear(D, L),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        self.w = nn.Linear(L, 1)

    def forward(self, H):
        A_raw = self.w(self.tanhV(H) * self.sigmU(H))
        A_norm = F.softmax(A_raw, dim=0)
        assert abs(A_norm.sum() - 1) < 1e-3
        return A_norm

class COCOH(nn.Module):
    def __init__(self, input_dims={'he_5x': 768, 'he_10x': 768, 'he_20x': 768, 'krt13': 768}, 
                 disease_dim=1, hidden_dim=256, dropout=0.25, n_classes=2):
        super(COCOH, self).__init__()
        
        self.modalities = list(input_dims.keys())
        
        self.inst_level_fcs = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for mod, input_dim in input_dims.items()
        })
        
        self.modal_attns = nn.ModuleDict({
            mod: GatedAttentionThSig(L=hidden_dim, D=hidden_dim)
            for mod in input_dims.keys()
        })
        
        self.disease_fc = nn.Sequential(
            nn.Linear(disease_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.modal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(input_dims), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, X_dict, disease):
        z_modalities = []
        A_norms = {}
        
        for mod in self.modalities:
            H_inst = self.inst_level_fcs[mod](X_dict[mod])
            A_norm = self.modal_attns[mod](H_inst)
            z_mod = torch.sum(A_norm * H_inst, dim=0)
            z_modalities.append(z_mod)
            A_norms[mod] = A_norm
        
        z_wsi = self.modal_fusion(torch.cat(z_modalities, dim=0))
        z_disease = self.disease_fc(disease.unsqueeze(0))
        z_combined = torch.cat([z_wsi, z_disease.squeeze(0)], dim=0)
        
        logits = self.classifier(z_combined).unsqueeze(dim=0)
        return logits, A_norms

class MultiMILData(Dataset):
    def __init__(self, feats_dirpaths, csv_fpath, which_split='train', which_labelcol='MT', is_external=False):
        self.feats_dirpaths = feats_dirpaths
        self.csv = pd.read_csv(csv_fpath)
        self.which_labelcol = which_labelcol
        self.is_external = is_external
        
        if self.is_external:
            self.csv_split = self.csv
        else:
            self.csv_split = self.csv[self.csv['split'] == which_split]

    def __getitem__(self, index):
        features_dict = {}
        row = self.csv_split.iloc[index]
        
        features_dict['he_5x'] = torch.load(
            os.path.join(self.feats_dirpaths['he_5x'], row['he_bag_id'] + '.pt'),
            weights_only=True
        )
        features_dict['he_10x'] = torch.load(
            os.path.join(self.feats_dirpaths['he_10x'], row['he_bag_id'] + '.pt'),
            weights_only=True
        )
        features_dict['he_20x'] = torch.load(
            os.path.join(self.feats_dirpaths['he_20x'], row['he_bag_id'] + '.pt'),
            weights_only=True
        )
        
        if pd.notna(row['krt_bag_id']):
            features_dict['krt13'] = torch.load(
                os.path.join(self.feats_dirpaths['krt13'], row['krt_bag_id'] + '.pt'),
                weights_only=True
            )
        else:
            features_dict['krt13'] = torch.zeros_like(features_dict['he_5x'])
        
        label = row[self.which_labelcol]
        disease = torch.tensor(row['disease'], dtype=torch.float32)
        
        return features_dict, disease, label, row['he_bag_id']

    def __len__(self):
        return self.csv_split.shape[0]

def calculate_metrics(y_true, y_prob, disease_values, prefix=''):
    metrics = {}
    
#ALL METRICS FOR OPMD
    metrics[f'{prefix}acc'] = sklearn.metrics.balanced_accuracy_score(y_true, y_prob > 0.5)
    if len(np.unique(y_true)) > 1:
        metrics[f'{prefix}auc'] = sklearn.metrics.roc_auc_score(y_true, y_prob)
        metrics[f'{prefix}auprc'] = sklearn.metrics.average_precision_score(y_true, y_prob)
    metrics[f'{prefix}brier'] = sklearn.metrics.brier_score_loss(y_true, y_prob)
    
#STRATIFIED METRICS FOR FAIRNESS
    for disease_val in [1, 2]:
        mask = (disease_values == disease_val)
        if sum(mask) == 0:
            continue
            
        disease_name = 'leukoplakia' if disease_val == 1 else 'lichenoid'
        y_true_sub = y_true[mask]
        y_prob_sub = y_prob[mask]
        
        metrics[f'{prefix}{disease_name}_acc'] = sklearn.metrics.balanced_accuracy_score(y_true_sub, y_prob_sub > 0.5)
        if len(np.unique(y_true_sub)) > 1:
            metrics[f'{prefix}{disease_name}_auc'] = sklearn.metrics.roc_auc_score(y_true_sub, y_prob_sub)
            metrics[f'{prefix}{disease_name}_auprc'] = sklearn.metrics.average_precision_score(y_true_sub, y_prob_sub)
        metrics[f'{prefix}{disease_name}_brier'] = sklearn.metrics.brier_score_loss(y_true_sub, y_prob_sub)
    
    return metrics

def train_val_test_extloop(epoch, model, loader, optimizer=None, loss_fn=None, 
                    split='train', device='cuda', verbose=1, print_every=300,
                    save_predictions=False):
    model.train() if split == 'train' else model.eval()
    total_loss = 0.0
    
#METRICS
    all_probs = []
    all_labels = []
    all_diseases = []
    
#SAVE PREDICTIONS
    wsi_names = []
    disease_values = []
    predictions = []
    true_labels = []

    for batch_idx, (X_dict, disease, label, wsi_name) in enumerate(loader):
        X_dict = {mod: X[0].to(device) for mod, X in X_dict.items()}
        disease, label = disease.to(device), label.to(device)

        if split == 'train':
            logits, _ = model(X_dict, disease)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        else:
            with torch.no_grad():
                logits, _ = model(X_dict, disease)
            loss = loss_fn(logits, label)

        total_loss += loss.item()
        
        logits = logits.squeeze(0)  
        probs = torch.softmax(logits, dim=-1)[1].item() 
        
        all_probs.append(probs)
        all_labels.append(label.item())
        all_diseases.append(disease.item())
        
        if save_predictions:
            wsi_names.append(wsi_name[0])
            disease_values.append(disease.item())
            true_labels.append(label.item())
            predictions.append(probs)

        if ((batch_idx + 1) % print_every == 0) and (verbose >= 2):
            print(f'Epoch {epoch}:\t Batch {batch_idx}\t Avg Loss: {total_loss / (batch_idx+1):.04f}')

#Calculating all metrics
    log_dict = {'loss': total_loss / len(loader)}
    log_dict.update(calculate_metrics(np.array(all_labels), np.array(all_probs), 
                                     np.array(all_diseases), prefix=f'{split}_'))
    
    if verbose >= 1:
        print(f'\n### ({split.capitalize()} Summary) ###')
        for metric, value in log_dict.items():
            print(f'{metric:20}: {value:.4f}')
    
    if save_predictions:
        pred_df = pd.DataFrame({
            'WSI_Name': wsi_names,
            'MT_Label': true_labels,
            'Disease_Covariate': disease_values,
            'Probability': predictions,
            'Predicted_Class': (np.array(predictions) > 0.5).astype(int)
        })
        return log_dict, pred_df
    else:
        return log_dict

def main():
    torch.manual_seed(2023)
    
#MIL DATASET PATHS
    feats_dirpaths = {
        'he_5x': [PATH],
        'he_10x': [PATH],
        'he_20x': [PATH],
        'krt13': [PATH]'
    }
    csv_fpath = [PATH]'
    
    external_feats_dirpaths = {
        'he_5x': [PATH],
        'he_10x': [PATH],
        'he_20x': [PATH],
        'krt13': [PATH]'
    }
    external_csv_fpath = [PATH].csv'
    
#LOADER
    loader_kwargs = {'batch_size': 1, 'num_workers': 0, 'pin_memory': False}
    train_dataset = MultiMILData(feats_dirpaths, csv_fpath, 'train')
    val_dataset = MultiMILData(feats_dirpaths, csv_fpath, 'val')
    test_dataset = MultiMILData(feats_dirpaths, csv_fpath, 'test')
    external_dataset = MultiMILData(external_feats_dirpaths, external_csv_fpath, 'test', is_external=True)
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    external_loader = DataLoader(external_dataset, shuffle=False, **loader_kwargs)
    
#SET UP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = COCOH(
        input_dims={'he_5x': 768, 'he_10x': 768, 'he_20x': 768, 'krt13': 768},
        hidden_dim=256,
        dropout=0.25
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = FocalCE(alpha=0.75, gamma=2.0)
    
#TRAINING LOOP
    num_epochs, min_early_stopping, patience, counter = 100, 5, 5, 0
    highest_auprc, best_model = 0.0, None
    best_epoch = 0
    
    for epoch in range(num_epochs):
        train_log = train_val_test_extloop(epoch, model, train_loader, optimizer, loss_fn, 'train', device)
        val_log = train_val_test_extloop(epoch, model, val_loader, None, loss_fn, 'val', device)
        
        current_auprc = val_log['val_auprc']
        
        if epoch > min_early_stopping:
            if current_auprc > highest_auprc:
                print(f'New best AUPRC: {highest_auprc:.04f} -> {current_auprc:.04f}')
                highest_auprc = current_auprc
                best_model = copy.deepcopy(model)
                best_epoch = epoch
                counter = 0
            else:
                counter += 1
                print(f'No improvement for {counter}/{patience} epochs')
                
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
    
#VAL TEST EXT RESULTS
    model.load_state_dict(best_model.state_dict())
    test_log, test_pred_df = train_val_test_extloop(0, model, test_loader, None, loss_fn, 'test', device, save_predictions=True)
    val_log, val_pred_df = train_val_test_extloop(0, model, val_loader, None, loss_fn, 'val', device, save_predictions=True)
    external_log, external_pred_df = train_val_test_extloop(0, model, external_loader, None, loss_fn, 'external_test', device, save_predictions=True)
    
    print("\n=== Final Performance ===")
    print(f"Best Val AUPRC: {highest_auprc:.4f} (epoch {best_epoch})")
    
#SAVE RESULTS AS CSV
    with pd.ExcelWriter(r'[PATH].xlsx') as writer:
        val_pred_df.to_excel(writer, sheet_name='Validation', index=False)
        test_pred_df.to_excel(writer, sheet_name='Test', index=False)
        external_pred_df.to_excel(writer, sheet_name='External', index=False)
    
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'metrics': {
            'val': val_log,
            'test': test_log,
            'external': external_log
        }
    }, [PATH].ckpt')

if __name__ == '__main__':
    main()

