"""
示例7: 深度学习光谱分类 — 1D-CNN
需要 PyTorch
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")

np.random.seed(42)

# ============================================================
# 生成数据
# ============================================================
n_per_class = 100
n_wavelengths = 500

def make_spectrum(cls, noise=0.02):
    spec = np.random.normal(0, noise, n_wavelengths)
    centers = {0: [1200, 1700, 2900], 1: [1100, 1540, 3300], 2: [1300, 1600, 2200]}
    for c in centers[cls]:
        spec += np.random.uniform(0.5, 0.9) * np.exp(-0.5 * ((np.arange(n_wavelengths) - c) / np.random.uniform(15, 40)) ** 2)
    return spec

spectra = []
labels = []
for cls in range(3):
    for _ in range(n_per_class):
        spectra.append(make_spectrum(cls))
        labels.append(cls)
spectra = np.array(spectra, dtype=np.float32)
labels = np.array(labels)

print("=" * 60)
print("  1D-CNN光谱分类示例")
print("=" * 60)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    # 预处理
    from irspectoolkit.preprocessing.transform import snv
    spectra_snv = snv(spectra).astype(np.float32)

    # 划分
    X_train, X_test, y_train, y_test = train_test_split(
        spectra_snv, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # 转为PyTorch张量 (N, 1, L) — 1D卷积需要channel维度
    X_train_t = torch.tensor(X_train).unsqueeze(1)
    X_test_t = torch.tensor(X_test).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

    # 定义1D-CNN
    class CNN1D(nn.Module):
        def __init__(self, n_wavelengths, n_classes):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128 * 8, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    model = CNN1D(n_wavelengths, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(30):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30, Loss: {total_loss/len(train_loader):.4f}")

    # 评估
    model.eval()
    with torch.no_grad():
        pred = model(X_test_t).argmax(dim=1).numpy()
    accuracy = (pred == y_test).mean()
    print(f"\n测试准确率: {accuracy:.3f}")

except ImportError:
    print("\n需要安装 PyTorch: pip install torch")
    print("跳过深度学习示例")
