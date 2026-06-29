import os
import argparse
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

from pl_model import LitModel
from data.st_data import KPS_Geodesic_Dataset, NAMES2ID
from utils.metrics import get_cd, hungary_iou

# Ensure PyTorch safely loads the model checkpoints
torch.serialization.add_safe_globals([argparse.Namespace])

def evaluate_checkpoint(checkpoint_path, dataset):
    """
    Evaluates a single model checkpoint over the entire dataset.
    Returns the average Chamfer Distance and a dictionary of mIoU values per threshold.
    """
    print(f"\n[Evaluating] Loading checkpoint: {os.path.basename(checkpoint_path)}")
    model = LitModel.load_from_checkpoint(checkpoint_path).cuda()
    model.eval()

    mcd_list = []
    
    # Initialize the IoU thresholds (0.00 to 0.10)
    thresholds = [i * 0.01 for i in range(11)]
    hmiou_results = {key: [] for key in thresholds}

    # Run inference over the evaluation dataset (without 3D popups)
    for i in tqdm.tqdm(range(len(dataset)), desc="Evaluating models"):
        pc, heat, _ = dataset[i]

        pc = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).cuda()
        heat = torch.tensor(heat, dtype=torch.float32).unsqueeze(0).cuda()
        
        with torch.no_grad():
            pts, gts = model.infer(pc, heat)
            pts = pts.cpu().numpy()
            gts = gts.cpu().numpy()

        # Compute pair-wise distance matrix
        dists = cdist(gts, pts, metric='euclidean')
        
        # 1. Compute Chamfer Distance
        cd_val = get_cd(dists)
        mcd_list.append(cd_val)

        # 2. Compute Hungarian IoU for each threshold
        for thr in thresholds:
            hiou = hungary_iou(dists, thr)
            hmiou_results[thr].append(hiou)

    # Calculate final averages
    avg_mcd = np.mean(mcd_list)
    avg_hmiou = {thr: np.mean(hmiou_results[thr]) for thr in thresholds}
    
    print(f"--> Done. Avg CD: {avg_mcd:.5f} | Max IoU (at 0.10): {avg_hmiou[0.10]:.4f}")
    return avg_mcd, avg_hmiou


def plot_results(experiments_data, output_dir="plots"):
    """
    Generates two scientific plots:
    1. Line plot: Hungarian mIoU over different tolerance thresholds.
    2. Bar plot: Mean Chamfer Distance (MCD) across different models.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    thresholds = [i * 0.01 for i in range(11)]
    
    # ---- Graph 1: Hungarian mIoU Curve ----
    plt.figure(figsize=(8, 5.5))
    for label, data in experiments_data.items():
        iou_values = [data["iou"][thr] for thr in thresholds]
        plt.plot(thresholds, iou_values, marker='o', linewidth=2, label=label)
        
    plt.title("Hungarian mIoU vs. Tolerance Threshold", fontsize=14, fontweight='bold')
    plt.xlabel("Tolerance Threshold", fontsize=12)
    plt.ylabel("Mean Intersection over Union (mIoU)", fontsize=12)
    plt.xticks(thresholds)
    plt.ylim(0, 1.05)
    plt.legend(title="Data Scale / Model", fontsize=10)
    plt.tight_layout()
    
    iou_plot_path = os.path.join(output_dir, "iou_threshold_comparison.png")
    plt.savefig(iou_plot_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] mIoU comparison graph saved to: {iou_plot_path}")

    # ---- Graph 2: Mean Chamfer Distance (MCD) Bar Chart ----
    plt.figure(figsize=(7, 5))
    labels = list(experiments_data.keys())
    cd_values = [experiments_data[lbl]["cd"] for lbl in labels]
    
    # Use a nice seaborn color palette for the bars
    colors = sns.color_palette("Blues_r", len(labels))
    bars = plt.bar(labels, cd_values, color=colors, edgecolor='grey', width=0.5)
    
    # Add exact value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + (max(cd_values)*0.01),
                 f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title("Chamfer Distance (CD) Comparison", fontsize=14, fontweight='bold')
    plt.ylabel("Chamfer Distance (Lower is Better)", fontsize=12)
    plt.xlabel("Data Scale / Model Configuration", fontsize=12)
    # Give some headspace for the text labels
    plt.ylim(0, max(cd_values) * 1.15) 
    plt.tight_layout()
    
    cd_plot_path = os.path.join(output_dir, "cd_comparison.png")
    plt.savefig(cd_plot_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] CD comparison graph saved to: {cd_plot_path}")


if __name__ == "__main__":
    # 1. Define all your trained checkpoints (Adjust paths to your real weight files)
    # The dictionary key will be used as the label inside the graphs.
    #checkpoints_to_compare = {
    #    "100%": "runs/keypoint_saliency/airplane_split100/checkpoints/last.ckpt",
    #    "50%": "runs/keypoint_saliency/airplane_split50/checkpoints/last.ckpt",
    #    "25%": "runs/keypoint_saliency/airplane_split25/checkpoints/last.ckpt",
    #    "10%": "runs/keypoint_saliency/airplane_split10/checkpoints/last.ckpt",
    #    "5%": "runs/keypoint_saliency/airplane_split25/checkpoints/epoch=4-step=445.ckpt",
    #    "2%": "runs/keypoint_saliency/airplane_split2/checkpoints/last.ckpt",
    #}
    checkpoints_to_compare = {
        "airplane": "runs/keypoint_saliency/airplane/checkpoints/last.ckpt",
        "chair": "runs/keypoint_saliency/chair/checkpoints/last.ckpt",
        "bed": "runs/keypoint_saliency/bed/checkpoints/last.ckpt",
        "cap": "runs/keypoint_saliency/cap/checkpoints/last.ckpt",
        "helmet": "runs/keypoint_saliency/helmet/checkpoints/last.ckpt",
        "skateboard": "runs/keypoint_saliency/skateboard/checkpoints/last.ckpt",
    }

    results_cache = {}
    
    for label, ckpt_path in checkpoints_to_compare.items():
        if not os.path.exists(ckpt_path):
            print(f"[Skipping] Checkpoint not found: {ckpt_path}")
            continue
        
        # Load the checkpoint to extract its specific class hyperparameters
        print(f"\n[Loading Hparams] Preparing dataset for {label}...")
        current_model = LitModel.load_from_checkpoint(ckpt_path)
        current_args = current_model.hparams.args
        
        # Override the split root path to use the relative setup
        current_args.split_root = '../KeypointNet/splits'
        test_file = 'test.txt'
        
        # Initialize the test dataset specifically for this object class
        print(f"Initializing evaluation dataset for class: {current_args.class_name}")
        current_dataset = KPS_Geodesic_Dataset(current_args, test_file, False)
        
        # Evaluate the checkpoint with its matching dataset
        avg_cd, avg_hmiou = evaluate_checkpoint(ckpt_path, current_dataset)
        
        results_cache[label] = {
            "cd": avg_cd,
            "iou": avg_hmiou
        }

    # 4. Generate the scientific plots
    if results_cache:
        plot_results(results_cache, output_dir="evaluation_plots")
    else:
        print("Error: No checkpoints were successfully evaluated.")