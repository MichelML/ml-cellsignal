import numpy as np
import pandas as pd
from scipy import misc
import torch
from tqdm import tqdm_notebook
from collections import Counter

from sklearn.model_selection import train_test_split

def eval_model_per_cell(model, loader, file_path, path_data, sub_df, device='cuda', sub_file='/artifacts/submission.csv'):
    model.load_state_dict(torch.load(file_path))
    model.eval()

    with torch.no_grad():
        preds = np.empty(0)
        all_preds = []
        for x1, x2, _ in tqdm_notebook(loader): 
            x1 = x1.to(device)
            x2 = x2.to(device)
            output = model(x1,x2)
            
            idx = output.max(dim=-1)[1].cpu().numpy()
            preds = np.append(preds, idx, axis=0)
            
            idx = output.cpu().numpy()
            all_preds.append(idx[0])

    sub_df['sirna'] = preds.astype(int)
    sub_df.to_csv(sub_file, index=False, columns=['id_code','sirna'])
    
    return all_preds, preds

def eval_model(model, loader, file_path, path_data, device='cuda', sub_file='/artifacts/submission.csv'):
    model.load_state_dict(torch.load(file_path))
    model.eval()

    with torch.no_grad():
        preds = np.empty(0)
        all_preds = []
        for x1, x2, _ in tqdm_notebook(loader): 
            x1 = x1.to(device)
            x2 = x2.to(device)
            output = model(x1,x2)
            
            idx = output.max(dim=-1)[1].cpu().numpy()
            preds = np.append(preds, idx, axis=0)
            
            idx = output.cpu().numpy()
            all_preds.append(idx[0])

    submission = pd.read_csv(path_data + '/test.csv')
    submission['sirna'] = preds.astype(int)
    submission.to_csv(sub_file, index=False, columns=['id_code','sirna'])
    
    return all_preds, preds
    
def eval_model_10(model, loader, file_path, path_data, device='cuda'):
    model.load_state_dict(torch.load(file_path))
    model.eval()
    with torch.no_grad():
        preds = np.empty(0)
        for image_pairs, _ in tqdm_notebook(loader):
            idx_counter = Counter()
            for image_pair in image_pairs:
                x1, x2 = image_pair
                x1 = x1.to(device)
                x2 = x2.to(device)
                output = model(x1,x2)
                idx = output.max(dim=-1)[1].cpu().numpy()
                idx_counter.update(idx)
            
            preds = np.append(preds, [idx_counter.most_common(1)[0][0]], axis=0)

    submission = pd.read_csv(path_data + '/test.csv')
    submission['sirna'] = preds.astype(int)
    submission.to_csv(f'submission.csv', index=False, columns=['id_code','sirna'])
