import torch
import numpy as np
import pandas as pd
import requests
import zipfile
import io
from io import StringIO
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_ford_a():
    """Baixa e processa o FordA (Sinais Industriais Ruidosos)."""
    print("⏳ Baixando FordA...")
    urls = {
        "train": "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TRAIN.tsv",
        "test": "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/FordA_TEST.tsv"
    }
    
    try:
        def get_data(url):
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            d = np.loadtxt(StringIO(r.text), delimiter="\t")
            return d[:, 1:], d[:, 0].astype(int)

        X_tr, y_tr = get_data(urls["train"])
        X_te, y_te = get_data(urls["test"])
        
        X = np.concatenate([X_tr, X_te]).astype(np.float32)
        y = np.concatenate([y_tr, y_te])
        y = np.where(y == -1, 0, 1) # Ajustar labels para 0/1
        
        return X, y, "FordA (Industrial)"
    except Exception as e:
        print(f"❌ Erro FordA: {e}")
        return None, None, None

def load_ecg5000():
    """Baixa e processa o ECG5000 (Sinais Cardíacos Alinhados)."""
    print("⏳ Baixando ECG5000...")
    url = "https://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"
    
    try:
        df = pd.read_csv(url, header=None)
        data = df.values.astype(np.float32)
        X = data[:, :-1]
        y = data[:, -1].astype(int)
        return X, y, "ECG5000 (Cardio)"
    except Exception as e:
        print(f"❌ Erro ECG: {e}")
        return None, None, None

def augment(x, device):
    """Aplica Jitter e Scaling (Invariância)."""
    noise = torch.randn_like(x, device=device) * 0.1
    scale = torch.randn(x.size(0), 1, 1, device=device) * 0.2 + 1.0
    return (x * scale) + noise

def get_loaders(X, y, batch_size=64):
    # Split 80/20
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaler
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    
    # Tensors
    train_loader = DataLoader(
        torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    
    return train_loader, X_tr, X_te, y_tr, y_te