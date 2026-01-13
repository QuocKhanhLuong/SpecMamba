"""
Deep Error Analysis Script
Phân tích chi tiết lỗi segmentation để xác định nguyên nhân:
1. Dice/Recall/Precision cho từng Class và từng Slice
2. Biểu đồ Dice vs. Volume để chứng minh Small Object Problem
3. Top 5 ca tệ nhất để debug trực quan
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse

from data.acdc_dataset import ACDCDataset2D
from training.test_advanced_arch import HRNetAdvanced

CLASS_NAMES = {1: "RV", 2: "MYO", 3: "LV"}


def compute_detailed_metrics(pred, target, num_classes=4):
    """Tính chi tiết Dice, Precision, Recall, Volume cho từng class"""
    results = {}
    for c in range(1, num_classes):  # Bỏ background
        p = (pred == c).float()
        t = (target == c).float()
        
        intersection = (p * t).sum().item()
        p_sum = p.sum().item()
        t_sum = t.sum().item()
        
        # Dice
        dice = (2 * intersection) / (p_sum + t_sum + 1e-6)
        
        # Precision (Chống dư nhãn) = TP / (TP + FP)
        precision = intersection / (p_sum + 1e-6)
        
        # Recall (Chống mất nhãn) = TP / (TP + FN)
        recall = intersection / (t_sum + 1e-6)
        
        # Volume (Kích thước vật thể)
        volume = t_sum
        
        results[c] = {
            "Dice": dice,
            "Precision": precision,
            "Recall": recall,
            "Volume": volume
        }
    return results


def classify_error(m):
    """Phân loại lỗi dựa trên metrics"""
    if m["Dice"] >= 0.85:
        return "Good"
    if m["Volume"] == 0:
        return "Empty_GT"  # GT rỗng
    if m["Recall"] < 0.5 and m["Precision"] > 0.8:
        return "Missing (Mất nhãn)"
    if m["Precision"] < 0.5 and m["Recall"] > 0.8:
        return "Over-seg (Dư nhãn)"
    if m["Volume"] < 300:
        return "Small Object"
    return "Boundary/Mismatch"


def plot_analysis(df, save_path="error_analysis.png"):
    """Vẽ biểu đồ phân tích"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Dice vs Volume (Scatter plot)
    ax1 = axes[0, 0]
    for cls in df["Class"].unique():
        mask = df["Class"] == cls
        ax1.scatter(df.loc[mask, "Volume"], df.loc[mask, "Dice"], 
                    alpha=0.5, label=cls, s=20)
    ax1.set_xscale("log")
    ax1.set_title("Dice Score vs. Object Size (Log Scale)")
    ax1.set_xlabel("Object Volume (pixels)")
    ax1.set_ylabel("Dice Score")
    ax1.axhline(0.9, color='r', linestyle='--', label='Dice=0.9')
    ax1.axhline(0.85, color='orange', linestyle='--', label='Dice=0.85')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision vs Recall
    ax2 = axes[0, 1]
    for cls in df["Class"].unique():
        mask = df["Class"] == cls
        ax2.scatter(df.loc[mask, "Recall"], df.loc[mask, "Precision"], 
                    alpha=0.5, label=cls, s=20)
    ax2.set_title("Precision vs. Recall Trade-off")
    ax2.set_xlabel("Recall (Sensitivity)")
    ax2.set_ylabel("Precision")
    ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Dice distribution per class
    ax3 = axes[1, 0]
    df.boxplot(column="Dice", by="Class", ax=ax3)
    ax3.set_title("Dice Distribution by Class")
    ax3.set_xlabel("Class")
    ax3.set_ylabel("Dice Score")
    plt.suptitle('')  # Remove auto title
    
    # 4. Error type distribution
    ax4 = axes[1, 1]
    error_counts = df["Error_Type"].value_counts()
    colors = {'Good': 'green', 'Missing (Mất nhãn)': 'red', 
              'Over-seg (Dư nhãn)': 'orange', 'Small Object': 'purple',
              'Boundary/Mismatch': 'blue', 'Empty_GT': 'gray'}
    ax4.bar(error_counts.index, error_counts.values, 
            color=[colors.get(x, 'gray') for x in error_counts.index])
    ax4.set_title("Error Type Distribution")
    ax4.set_xlabel("Error Type")
    ax4.set_ylabel("Count")
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n>>> Saved analysis plot: {save_path}")


def save_worst_cases(df, img_list, pred_list, target_list, save_dir="analysis/worst_cases"):
    """Lưu hình ảnh các ca tệ nhất để debug"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Lấy 10 ca có Dice thấp nhất (không phải Empty_GT)
    bad_df = df[df["Error_Type"] != "Empty_GT"].nsmallest(10, "Dice")
    
    print(f"\n>>> Top 10 worst cases:")
    print(bad_df[["Sample_ID", "Class", "Dice", "Precision", "Recall", "Error_Type"]])
    
    unique_samples = bad_df["Sample_ID"].unique()[:5]
    
    for idx, sample_id in enumerate(unique_samples):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        img = img_list[sample_id]
        pred = pred_list[sample_id]
        target = target_list[sample_id]
        
        # Input image
        if img.shape[0] == 3:
            axes[0].imshow(img.transpose(1, 2, 0)[:, :, 0], cmap='gray')
        else:
            axes[0].imshow(img[0], cmap='gray')
        axes[0].set_title("Input")
        axes[0].axis('off')
        
        # Prediction
        axes[1].imshow(pred, cmap='viridis', vmin=0, vmax=3)
        axes[1].set_title("Prediction")
        axes[1].axis('off')
        
        # Ground Truth
        axes[2].imshow(target, cmap='viridis', vmin=0, vmax=3)
        axes[2].set_title("Ground Truth")
        axes[2].axis('off')
        
        plt.suptitle(f"Sample {sample_id} - Worst Case #{idx+1}")
        plt.savefig(os.path.join(save_dir, f"worst_{idx+1}_sample_{sample_id}.png"), 
                    dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"\n>>> Saved {len(unique_samples)} worst case images to '{save_dir}/'")


def analyze(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Data
    print(">>> Loading data...")
    dataset = ACDCDataset2D(args.data_dir, in_channels=3)
    
    # Split (same as training)
    num_vols = len(dataset.vol_paths)
    vol_indices = list(range(num_vols))
    np.random.seed(42)
    np.random.shuffle(vol_indices)
    split = int(num_vols * 0.8)
    val_vols = set(vol_indices[split:])
    val_indices = [i for i, (v, s) in enumerate(dataset.index_map) if v in val_vols]
    
    val_ds = torch.utils.data.Subset(dataset, val_indices)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    print(f">>> Validation samples: {len(val_ds)}")
    
    # Load Model
    print(f">>> Loading model from: {args.checkpoint}")
    
    # Infer config from checkpoint
    use_pointrend = 'pointrend' in args.checkpoint.lower()
    
    # Try to detect block counts from checkpoint keys
    stage2_depth = args.stage2_blocks
    stage3_depth = args.stage3_blocks  
    stage4_depth = args.stage4_blocks
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        
        # Auto-detect block counts from checkpoint keys
        max_s2, max_s3, max_s4 = 0, 0, 0
        for key in checkpoint.keys():
            if key.startswith('stage2.branches.0.'):
                idx = int(key.split('.')[2]) + 1 if key.split('.')[2].isdigit() else 0
                max_s2 = max(max_s2, int(key.split('.')[3]) + 1) if key.split('.')[3].isdigit() else max_s2
            elif key.startswith('stage3.branches.0.'):
                max_s3 = max(max_s3, int(key.split('.')[3]) + 1) if key.split('.')[3].isdigit() else max_s3
            elif key.startswith('stage4.branches.0.'):
                max_s4 = max(max_s4, int(key.split('.')[3]) + 1) if key.split('.')[3].isdigit() else max_s4
        
        if max_s2 > 0: stage2_depth = max_s2
        if max_s3 > 0: stage3_depth = max_s3
        if max_s4 > 0: stage4_depth = max_s4
        
        print(f">>> Detected config: stage2={stage2_depth}, stage3={stage3_depth}, stage4={stage4_depth}")
    else:
        checkpoint = None
        print(f">>> WARNING: Checkpoint not found, using random weights!")
    
    model = HRNetAdvanced(
        in_channels=3,
        base_channels=64,
        num_classes=4,
        stage_configs=[
            {'blocks': ['dcn'] * stage2_depth},
            {'blocks': ['dcn'] * stage3_depth},
            {'blocks': ['dcn'] * stage4_depth}
        ],
        use_pointrend=use_pointrend
    ).to(device)
    
    if checkpoint:
        model.load_state_dict(checkpoint, strict=False)
        print(">>> Checkpoint loaded successfully!")
    
    model.eval()
    
    # Analyze
    records = []
    img_list, pred_list, target_list = [], [], []
    
    print("\n>>> Scanning for errors...")
    with torch.no_grad():
        for i, (img, target) in enumerate(tqdm(val_loader)):
            img_np = img.squeeze(0).cpu().numpy()
            target_np = target.squeeze(0).cpu().numpy()
            
            img, target = img.to(device), target.to(device)
            
            out = model(img)['output']
            pred = out.argmax(dim=1)
            pred_np = pred.squeeze(0).cpu().numpy()
            
            # Store for visualization
            img_list.append(img_np)
            pred_list.append(pred_np)
            target_list.append(target_np)
            
            # Compute metrics
            metrics = compute_detailed_metrics(pred, target)
            
            for c, m in metrics.items():
                records.append({
                    "Sample_ID": i,
                    "Class": CLASS_NAMES[c],
                    "Dice": m["Dice"],
                    "Precision": m["Precision"],
                    "Recall": m["Recall"],
                    "Volume": m["Volume"],
                    "Error_Type": classify_error(m)
                })
    
    df = pd.DataFrame(records)
    
    # Print Report
    print("\n" + "="*60)
    print("FAILURE ANALYSIS REPORT")
    print("="*60)
    
    # 1. Overall metrics per class
    print("\n1. Per-Class Performance:")
    print("-"*40)
    class_summary = df.groupby("Class").agg({
        "Dice": ["mean", "std", "min"],
        "Precision": "mean",
        "Recall": "mean"
    }).round(4)
    print(class_summary)
    
    # 2. Performance by size
    print("\n2. Performance by Object Size:")
    print("-"*40)
    df_nonzero = df[df["Volume"] > 0]
    df_nonzero["Size_Bin"] = pd.qcut(df_nonzero["Volume"], q=4, labels=["Tiny", "Small", "Medium", "Large"])
    size_perf = df_nonzero.groupby("Size_Bin")["Dice"].agg(["mean", "std", "count"]).round(4)
    print(size_perf)
    
    # 3. Error type distribution
    print("\n3. Error Type Distribution:")
    print("-"*40)
    error_dist = df["Error_Type"].value_counts()
    for err_type, count in error_dist.items():
        pct = count / len(df) * 100
        print(f"  {err_type:<25}: {count:>5} ({pct:>5.1f}%)")
    
    # 4. Bad cases summary
    print("\n4. Problematic Cases (Dice < 0.7):")
    print("-"*40)
    bad_cases = df[df["Dice"] < 0.7]
    if len(bad_cases) > 0:
        bad_by_class = bad_cases.groupby("Class").size()
        print(bad_by_class)
        print(f"\nTotal bad cases: {len(bad_cases)} ({len(bad_cases)/len(df)*100:.1f}%)")
    else:
        print("  No cases with Dice < 0.7!")
    
    # 5. Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    # Analyze main issue
    main_error = error_dist.idxmax() if "Good" not in error_dist.index or error_dist.idxmax() != "Good" else error_dist.drop("Good").idxmax() if len(error_dist) > 1 else None
    
    if main_error == "Missing (Mất nhãn)":
        print("""
>>> ISSUE: Model thiếu Recall - bỏ sót nhiều vùng
>>> SOLUTIONS:
    1. Tăng trọng số class trong Loss (focal_gamma)
    2. Thử Tversky Loss với beta > 0.5
    3. Giảm threshold prediction xuống 0.3-0.4
""")
    elif main_error == "Over-seg (Dư nhãn)":
        print("""
>>> ISSUE: Model dư Precision - tô bừa nhiều vùng sai
>>> SOLUTIONS:
    1. Tăng cường DoG preprocessing
    2. Thêm regularization (weight_decay)
    3. Dùng stronger augmentation
""")
    elif main_error == "Small Object":
        print("""
>>> ISSUE: Small Object Problem - khối u nhỏ bị bỏ sót
>>> SOLUTIONS:
    1. Dùng Generalized Dice Loss hoặc Focal Loss
    2. Tăng độ phân giải input (256x256 hoặc 288x288)
    3. Thêm PointRend nếu chưa có
""")
    elif main_error == "Boundary/Mismatch":
        print("""
>>> ISSUE: Boundary Error - viền không chính xác
>>> SOLUTIONS:
    1. Tăng Boundary Loss weight
    2. Dùng PointRend
    3. Thêm CRF post-processing
""")
    else:
        print(">>> Model đang hoạt động tốt! Tiếp tục fine-tune.")
    
    # Save plots
    os.makedirs("analysis", exist_ok=True)
    plot_analysis(df, save_path="analysis/error_analysis.png")
    
    # Save worst cases
    save_worst_cases(df, img_list, pred_list, target_list)
    
    # Save CSV
    df.to_csv("analysis/detailed_metrics.csv", index=False)
    print("\n>>> Saved detailed metrics: analysis/detailed_metrics.csv")


def main():
    parser = argparse.ArgumentParser(description="Deep Error Analysis")
    parser.add_argument('--data_dir', type=str, default='preprocessed_data/ACDC/training')
    parser.add_argument('--checkpoint', type=str, 
                        default='weights/advanced_asymmetric_dcn___pointrend_best.pt')
    parser.add_argument('--stage2_blocks', type=int, default=2, help='Default blocks in stage2')
    parser.add_argument('--stage3_blocks', type=int, default=4, help='Default blocks in stage3')
    parser.add_argument('--stage4_blocks', type=int, default=6, help='Default blocks in stage4')
    args = parser.parse_args()
    
    analyze(args)


if __name__ == "__main__":
    main()
