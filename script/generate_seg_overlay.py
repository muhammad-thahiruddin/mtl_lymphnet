import os
import sys
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# PATH FIX TO ENABLE IMPORTING 'src'
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), "src")
sys.path.insert(0, src_dir)

# ==========================================
# MODULE IMPORTS
# ==========================================
from mtl_lymphnet.config import Config
from mtl_lymphnet.models.mtl_unet import MTLUNet
from mtl_lymphnet.data.inventory import build_dataset_inventory
from mtl_lymphnet.data.dataset import get_eval_loader 
from mtl_lymphnet.data.splits import make_nested_kfold_splits, apply_split

# ==========================================
# 1. INFERENCE FUNCTION
# ==========================================
def calculate_dice(pred, target, smooth=1e-5):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def run_inference(model, eval_loader, device='cuda', save_path='inference_results.pkl'):
    model.eval()
    model.to(device)
    
    results = {'benign': [], 'malignant': []}
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Inference Process"):
            images = batch["image"].to(device)
            masks_gt = batch["mask"].to(device)
            labels = batch["label"].to(device)
            
            seg_logits, _ = model(images)
            
            seg_probs = torch.sigmoid(seg_logits)
            seg_preds = (seg_probs > 0.5).float()
            
            images_np = images.cpu().numpy()
            masks_gt_np = masks_gt.cpu().numpy()
            seg_preds_np = seg_preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            for i in range(images.size(0)):
                dice = calculate_dice(seg_preds_np[i], masks_gt_np[i])
                
                data_dict = {
                    'image': images_np[i],
                    'mask_gt': masks_gt_np[i, 0],
                    'mask_pred': seg_preds_np[i, 0],
                    'dice_score': dice
                }
                
                if labels_np[i] == 0:
                    results['benign'].append(data_dict)
                else:
                    results['malignant'].append(data_dict)
                    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nInference results successfully saved to {save_path}")

# ==========================================
# 2. VISUALIZATION FUNCTION
# ==========================================
def unnormalize_image(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Restore image to original pixel range based on transformation mean/std."""
    # Assume img_tensor has dimensions (C, H, W)
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    
    img_unnorm = (img_tensor * std) + mean
    return np.clip(img_unnorm, 0, 1) # Ensure valid RGB [0, 1] range

def create_overlay(image_hwc, gt, pred):
    overlay = np.copy(image_hwc)
    overlay[(pred == 1) & (gt == 0)] = [1, 0, 0] # Red: FP
    overlay[(pred == 0) & (gt == 1)] = [0, 1, 0] # Green: FN
    overlay[(pred == 1) & (gt == 1)] = [1, 1, 0] # Yellow: TP
    
    return 0.5 * image_hwc + 0.5 * overlay

def plot_seg_results(pkl_path='inference_results.pkl', save_fig='seg_overlay_cases.png'):
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
        
    b_sorted = sorted(results['benign'], key=lambda x: x['dice_score'])
    m_sorted = sorted(results['malignant'], key=lambda x: x['dice_score'])
    
    def get_wmb(sorted_list):
        if not sorted_list: return [None, None, None]
        return [sorted_list[0], sorted_list[len(sorted_list)//2], sorted_list[-1]]
    
    all_cases = get_wmb(b_sorted) + get_wmb(m_sorted)
    titles = [
        "Benign: Worst", "Benign: Median", "Benign: Best",
        "Malignant: Worst", "Malignant: Median", "Malignant: Best"
    ]
    
    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 14))
    
    for col_idx, case in enumerate(all_cases):
        if case is None: continue
            
        img_chw, gt, pred, dice = case['image'], case['mask_gt'], case['mask_pred'], case['dice_score']
        
        # [FIX] Using un-normalize for proper visual representation
        img_unnorm_chw = unnormalize_image(img_chw)
        img_vis_hwc = np.transpose(img_unnorm_chw, (1, 2, 0))
        
        overlay = create_overlay(img_vis_hwc, gt, pred)
        
        axes[0, col_idx].imshow(img_vis_hwc)
        axes[0, col_idx].set_title(f"{titles[col_idx]}\nDice: {dice:.3f}")
        axes[0, col_idx].axis('off')
        
        axes[1, col_idx].imshow(gt, cmap='gray')
        if col_idx == 0: axes[1, col_idx].set_ylabel("Ground Truth", fontsize=12)
        axes[1, col_idx].axis('off')
        
        axes[2, col_idx].imshow(pred, cmap='gray')
        if col_idx == 0: axes[2, col_idx].set_ylabel("Prediction", fontsize=12)
        axes[2, col_idx].axis('off')
        
        axes[3, col_idx].imshow(overlay)
        if col_idx == 0: axes[3, col_idx].set_ylabel("Overlay", fontsize=12)
        axes[3, col_idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_fig, dpi=300, bbox_inches='tight')
    print(f"Visualization successfully saved to {save_fig}")


# ==========================================
# 3. EXECUTION BLOCK (MAIN)
# ==========================================
if __name__ == "__main__":
    cfg = Config(mode="full").apply_mode_defaults()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg.data_dir = os.path.join(os.path.dirname(current_dir), cfg.data_dir)
    cfg.output_dir = os.path.join(os.path.dirname(current_dir), cfg.output_dir)
    
    custom_out_dir = os.path.join(cfg.output_dir, "failure_cases_analysis")
    os.makedirs(custom_out_dir, exist_ok=True) 
    
    pkl_file = os.path.join(custom_out_dir, "inference_results.pkl")
    img_file = os.path.join(custom_out_dir, "seg_overlay_failure_cases.png")
    
    # [FIX] Logic to skip if pickle file already exists
    if os.path.exists(pkl_file):
        print(f"\n[INFO] Found old inference data at:\n-> {pkl_file}")
        print("Skipping inference process. (Delete this .pkl file if you want to re-run inference)")
    else:
        print(f"\n[INFO] Inference data not found. Starting new inference process...")
        df_full = build_dataset_inventory(cfg.data_dir)
        splits = make_nested_kfold_splits(
            df_full, 
            n_splits=cfg.n_folds, 
            val_fraction=cfg.val_fraction,
            label_col="label", 
            group_col=None, 
            seed=42
        )
        
        _, _, df_test = apply_split(df_full, splits[0]) 
        eval_loader = get_eval_loader(df_test, cfg)
        model = MTLUNet(backbone=cfg.backbone, pretrained=None)
        
        checkpoint_path = os.environ.get(
            "CHECKPOINT_PATH",
            os.path.join(cfg.output_dir, "checkpoints", "exp1_mtl_main_seed42_fold0.pt"),
        )
        
        if not os.path.exists(checkpoint_path):
            print(f"\n[ERROR] Model file not found at:\n-> {checkpoint_path}")
            sys.exit(1)
        
        print(f"\nSuccessfully found model at:\n-> {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        run_inference(model, eval_loader, device=device, save_path=pkl_file)
    
    # Execute plotting, whether from new results or loaded .pkl results
    print("\n[INFO] Creating visualization...")
    plot_seg_results(pkl_path=pkl_file, save_fig=img_file)