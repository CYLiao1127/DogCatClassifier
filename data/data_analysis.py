import os
import random
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


# 分析 train set 和 validation set 的貓狗比例
def analyze_dataset_distribution(full_dataset, train_indices, val_indices):
    # 獲取所有標籤
    all_labels = [full_dataset[i][1] for i in range(len(full_dataset))]
    
    # 分析 train set 標籤分布
    train_labels = [full_dataset[i][1] for i in train_indices]
    val_labels = [full_dataset[i][1] for i in val_indices]
    
    # 計算各類別數量 (假設 0=cat, 1=dog)
    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)
    full_counter = Counter(all_labels)
    
    # print統計結果
    print("=== Dataset Distribution Analysis ===")
    print(f"Full Dataset: {dict(full_counter)}")
    print(f"Train Set: {dict(train_counter)} (Total: {len(train_labels)})")
    print(f"Val Set: {dict(val_counter)} (Total: {len(val_labels)})")
    
    # 計算比例
    train_cat_ratio = train_counter[0] / len(train_labels) * 100
    train_dog_ratio = train_counter[1] / len(train_labels) * 100
    val_cat_ratio = val_counter[0] / len(val_labels) * 100
    val_dog_ratio = val_counter[1] / len(val_labels) * 100
    
    print(f"\nTrain Set Ratios: Cat {train_cat_ratio:.1f}%, Dog {train_dog_ratio:.1f}%")
    print(f"Val Set Ratios: Cat {val_cat_ratio:.1f}%, Dog {val_dog_ratio:.1f}%")
    
    return train_counter, val_counter, full_counter

# 視覺化分布
def plot_distribution_comparison(train_counter, val_counter, full_counter):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['#e74c3c', '#3498db']  # 紅色代表貓，藍色代表狗
    labels = ['Cat', 'Dog']
    
    # 準備數據
    datasets = [
        (full_counter, "Full Dataset", 0),
        (train_counter, "Training Set", 1), 
        (val_counter, "Validation Set", 2)
    ]
    
    for counter, title, idx in datasets:
        counts = [counter[0], counter[1]]  # [cat_count, dog_count]
        
        # 創建餅圖
        wedges, texts, autotexts = axes[idx].pie(counts, 
                                                labels=labels,
                                                autopct='%1.1f%%',
                                                colors=colors,
                                                explode=(0.05, 0.05),
                                                # shadow=True,
                                                startangle=90)
        
        # 設置標題
        axes[idx].set_title(f'{title}\n(Total: {sum(counts):,})', 
                           fontsize=12, fontweight='bold', pad=20)
        
        # 美化百分比文字
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
    
    plt.tight_layout()
    plt.savefig("train_val_distribution.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# 分析分布差異
def analyze_distribution_difference(train_counter, val_counter):
    train_total = sum(train_counter.values())
    val_total = sum(val_counter.values())
    
    train_cat_ratio = train_counter[0] / train_total
    val_cat_ratio = val_counter[0] / val_total
    
    ratio_diff = abs(train_cat_ratio - val_cat_ratio) * 100
    
    print(f"\n=== Distribution Difference Analysis ===")
    print(f"Cat ratio difference between train/val: {ratio_diff:.2f}%")