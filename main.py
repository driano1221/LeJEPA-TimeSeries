import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from src.data import load_ford_a, load_ecg5000, get_loaders, augment    
from src.model import LeJEPA
from sklearn.linear_model import LogisticRegression

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30 # Pode aumentar para 50 se tiver GPU
BATCH_SIZE = 64

def run_experiment(X, y, dataset_name, results_dir):
    print(f"\n{'='*40}\n🚀 Iniciando Experimento: {dataset_name}\n{'='*40}")
    
    # 1. Dados
    train_loader, X_tr, X_te, y_tr, y_te = get_loaders(X, y, BATCH_SIZE)
    input_len = X.shape[1]
    
    # 2. Modelo
    model = LeJEPA(input_len).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Treino
    print("   🏋️ Treinando LeJEPA...")
    model.train()
    losses = []
    for epoch in range(EPOCHS):
        ep_loss = 0
        for x_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            v1, v2 = augment(x_batch, DEVICE), augment(x_batch, DEVICE)
            
            optimizer.zero_grad()
            loss = model(v1, v2)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        avg_loss = ep_loss / len(train_loader)
        losses.append(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f"     Epoch {epoch+1}: Loss {avg_loss:.4f}")

    # Salvar Modelo Treinado (Checkpoint)
    ckpt_path = f"checkpoints/{dataset_name.split()[0].lower()}_model.pth"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"   💾 Modelo salvo em: {ckpt_path}")

    # 4. Avaliação (Few-Shot)
    print("   🔬 Avaliando (Linear Probe)...")
    model.eval()
    
    def get_features(data):
        # Processar em batches para não estourar memória
        features = []
        batch_size = 128
        for i in range(0, len(data), batch_size):
            batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32).unsqueeze(1).to(DEVICE)
            with torch.no_grad():
                features.append(model.encoder(batch).cpu().numpy())
        return np.concatenate(features)

    X_tr_feat = get_features(X_tr)
    X_te_feat = get_features(X_te)
    
    fractions = [0.01, 0.05, 0.1, 1.0]
    res_lejepa = []
    res_raw = []
    
    for frac in fractions:
        n = int(len(y_tr) * frac)
        if n < 10: n = 10
        idx = np.random.choice(len(y_tr), n, replace=False)
        
        # LeJEPA
        clf = LogisticRegression(max_iter=3000)
        clf.fit(X_tr_feat[idx], y_tr[idx])
        acc_l = clf.score(X_te_feat, y_te)
        res_lejepa.append(acc_l)
        
        # Baseline (Raw Data)
        clf_r = LogisticRegression(max_iter=3000)
        clf_r.fit(X_tr[idx], y_tr[idx])
        acc_r = clf_r.score(X_te, y_te)
        res_raw.append(acc_r)
        
        print(f"     {frac*100}% Labels -> LeJEPA: {acc_l*100:.1f}% | Raw: {acc_r*100:.1f}%")
        
    return fractions, res_lejepa, res_raw, losses

def main():
    # Criar pastas necessárias
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    datasets = [load_ecg5000(), load_ford_a()]
    all_metrics = {}

    plt.figure(figsize=(15, 6))

    for i, (X, y, name) in enumerate(datasets):
        if X is None: continue
        
        fracs, acc_l, acc_r, loss_hist = run_experiment(X, y, name, 'results')
        
        # Guardar métricas
        all_metrics[name] = {
            "fractions": fracs,
            "lejepa_accuracy": acc_l,
            "baseline_accuracy": acc_r,
            "training_loss": loss_hist
        }

        # Plotar
        plt.subplot(1, 2, i+1)
        plt.plot(fracs, acc_l, 'o-', label='LeJEPA', color='blue', linewidth=2)
        plt.plot(fracs, acc_r, 'x--', label='Baseline (Raw)', color='gray', alpha=0.7)
        plt.title(f"{name}\nData Efficiency")
        plt.xlabel("Fraction of Labels")
        plt.ylabel("Test Accuracy")
        plt.xscale('log')
        plt.xticks(fracs, [f"{int(f*100)}%" for f in fracs])
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Salvar Gráfico
    plt.savefig('results/benchmark_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Gráfico salvo em: results/benchmark_comparison.png")

    # Salvar Métricas em JSON
    with open('results/metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print("✅ Métricas salvas em: results/metrics.json")

if __name__ == "__main__":
    main()