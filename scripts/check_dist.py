import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

def check_distribution(dataset_path, save_path):
    print(f"Loading dataset: {dataset_path}")
    data_list = torch.load(dataset_path, weights_only=False)
    
    ys = []
    for data in data_list:
        # 兼容 y 是一维或二维的情况
        if data.y.numel() > 1:
            ys.append(data.y.view(-1)[0].item()) # 假设第一个是 kcat
        else:
            ys.append(data.y.item())
            
    ys = torch.tensor(ys)
    print(f"Data count: {len(ys)}")
    print(f"Min: {ys.min():.4f}, Max: {ys.max():.4f}")
    print(f"Mean: {ys.mean():.4f}, Std: {ys.std():.4f}")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(ys.numpy(), bins=50, kde=True)
    plt.title(f"Label Distribution in {os.path.basename(dataset_path)}")
    plt.xlabel("log10(kcat)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    print(f"Distribution plot saved to {save_path}")

if __name__ == "__main__":
    check_distribution("data/processed/kcat_full_1213.pt", "train_dist.png")
    check_distribution("data/processed/kcat_test_new.pt", "test_dist.png")

