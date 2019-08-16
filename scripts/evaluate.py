import numpy as np
import pandas as pd
import rxrxutils.rxrx.io as rio
from scipy import misc
import torch
from tqdm import tqdm_notebook

from sklearn.model_selection import train_test_split

def eval_model(model, loader, file_path, path_data, device='cuda'):
    model.load_state_dict(torch.load(file_path))
    model.eval()
    with torch.no_grad():
        preds = np.empty(0)
        for x1, x2, _ in tqdm_notebook(loader): 
            x1 = x1.to(device)
            x2 = x2.to(device)
            output = model(x1,x2)
            idx = output.max(dim=-1)[1].cpu().numpy()
            preds = np.append(preds, idx, axis=0)

    submission = pd.read_csv(path_data + '/test.csv')
    submission['sirna'] = preds.astype(int)
    submission.to_csv(f'submission.csv', index=False, columns=['id_code','sirna'])