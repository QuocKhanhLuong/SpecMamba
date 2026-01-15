
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from pathlib import Path
from scipy.ndimage import zoom
from datetime import datetime
from PIL import Image

# Add project root
# current file is in src/utils/visualize.py -> root is ../../
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CASES = -1  # -1 means all slices
IMG_SIZE = 224
NUM_CLASSES = 4  # ACDC: BG, RV, MYO, LV

# Filter specific patients (set to None to process all)
PATIENT_FILTER = ['patient103', 'patient104', 'patient105']

# Color map for segmentation (same for all models)
COLORS = {
    0: [0, 0, 0],       # Background - Black
    1: [255, 0, 0],     # RV - Red
    2: [0, 255, 0],     # MYO - Green
    3: [0, 0, 255],     # LV - Blue
}

CLASS_NAMES = {
    0: 'Background',
    1: 'RV',
    2: 'MYO',
    3: 'LV'
}

# Model weights paths
MODEL_CONFIGS = {
    'PIE-UNet': {
        'weights': PROJECT_ROOT / 'weights' / 'best_model_acdc_no_anatomical.pth',
        'type': 'pie_unet'
    },
    'Swin-Unet': {
        'weights': PROJECT_ROOT / 'comparison' / 'Swin-Unet' / 'acdc_out' / 'best_model.pth',
        'type': 'swin_unet'
    },
    'TransUNet': {
        'weights': PROJECT_ROOT / 'comparison' / 'TransUNet' / 'model' / 'TU_ACDC224' / 
                   'TU_pretrain_R50-ViT-B_16_skip3_epo150_bs24_224' / 'epoch_149.pth',
        'type': 'transunet'
    },
    'SwinUNETR': {
        'weights': PROJECT_ROOT / 'comparison' / 'SwinUNETR' / 'logs' / 'SwinUNETR_v2_ACDC224' / 'best_model.pth',
        'type': 'swinunetr'
    },
    'nnUNet': {
        'weights': PROJECT_ROOT / 'comparison' / 'nnUNet' / 'nnUNet_work_dir' / 'nnUNet_results' / 
                   'Dataset001_ACDC' / 'nnUNetTrainer_150epochs__nnUNetPlans__2d' / 'fold_0' / 'checkpoint_best.pth',
        'type': 'nnunet'
    },
    'UNet++': {
        'weights': PROJECT_ROOT / 'comparison' / 'pytorch-nested-unet' / 'models' / 'ACDC_NestedUNet' / 'model_best.pth',
        'type': 'unetpp'
    }
}

OUTPUT_DIR = PROJECT_ROOT / 'visualization_outputs' / 'model_comparison'


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

def load_pie_unet(weights_path):
    """Load PIE-UNet model."""
    # Assuming src.models.unet exists due to user input, but falling back to checking imports if needed
    try:
        from src.models.unet import RobustMedVFL_UNet
        model = RobustMedVFL_UNet(n_channels=5, n_classes=NUM_CLASSES, deep_supervision=True).to(DEVICE)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=False))
        model.eval()
        return model
    except ImportError:
        print("  ! PIE-UNet definition not found in src.models.unet")
        return None


def load_swin_unet(weights_path):
    """Load Swin-Unet model with CORRECT architecture (lite version)."""
    swin_unet_path = PROJECT_ROOT / 'comparison' / 'Swin-Unet'
    if not swin_unet_path.exists(): return None
    sys.path.insert(0, str(swin_unet_path))
    
    try:
        from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
        
        # CORRECT config from swin_tiny_patch4_window7_224_lite.yaml
        model_swin = SwinTransformerSys(
            img_size=IMG_SIZE,
            patch_size=4,
            in_chans=3,  # Trained with 3 channels
            num_classes=NUM_CLASSES,
            embed_dim=96,
            depths=[2, 2, 2, 2],  # FIXED: lite version uses [2,2,2,2] not [2,2,6,2]
            depths_decoder=[2, 2, 2, 1],  # FIXED: from config
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,  # FIXED: from config (was 0.1)
        ).to(DEVICE)
        
        # Load state dict
        state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Strip 'swin_unet.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('swin_unet.'):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        
        model_swin.load_state_dict(new_state_dict)
        model_swin.eval()
        return model_swin
    except Exception as e:
        print(f"  ! Error loading Swin-Unet: {e}")
        return None


def load_transunet(weights_path):
    """Load TransUNet model."""
    transunet_path = PROJECT_ROOT / 'comparison' / 'TransUNet'
    if not transunet_path.exists(): return None
    sys.path.insert(0, str(transunet_path))
    
    try:
        from networks.vit_seg_modeling import VisionTransformer as ViT_seg
        from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
        
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = NUM_CLASSES
        config_vit.n_skip = 3
        config_vit.patches.size = (16, 16)
        config_vit.patches.grid = (IMG_SIZE // 16, IMG_SIZE // 16)
        
        model = ViT_seg(config_vit, img_size=IMG_SIZE, num_classes=NUM_CLASSES).to(DEVICE)
        
        # Load state dict - handle different formats
        state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"  ! Error loading TransUNet: {e}")
        return None


def load_swinunetr(weights_path):
    """Load SwinUNETR model (MONAI implementation)."""
    try:
        from monai.networks.nets import SwinUNETR
        import inspect
        
        # Load state dict first to check if it was trained with v2
        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Check if model was trained with use_v2 (has layers*c keys)
        is_v2_model = any('layers1c' in k for k in state_dict.keys())
        print(f"  Detected use_v2={is_v2_model}")
        
        # Check if current MONAI supports use_v2 and img_size
        sig = inspect.signature(SwinUNETR.__init__)
        supports_v2 = 'use_v2' in sig.parameters
        supports_img_size = 'img_size' in sig.parameters
        
        if is_v2_model and not supports_v2:
            raise RuntimeError("MONAI version mismatch - need newer MONAI with use_v2 support")
        
        # Create model with appropriate parameters (MONAI 1.5+ doesn't have img_size)
        if supports_img_size:
            # Older MONAI
            model = SwinUNETR(
                img_size=(IMG_SIZE, IMG_SIZE),
                in_channels=1,
                out_channels=NUM_CLASSES,
                feature_size=48,
                use_v2=is_v2_model,
                spatial_dims=2,
            ).to(DEVICE)
        else:
            # Newer MONAI (1.5+) - no img_size parameter
            model = SwinUNETR(
                in_channels=1,
                out_channels=NUM_CLASSES,
                feature_size=48,
                use_v2=is_v2_model,
                spatial_dims=2,
            ).to(DEVICE)
        
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"  ! Error loading SwinUNETR: {e}")
        return None


def load_nnunet(weights_path):
    """Load nnUNet model from checkpoint using dynamic_network_architectures."""
    # nnUNet uses a special checkpoint format
    # Use weights_only=False for PyTorch 2.6+ compatibility with numpy objects
    try:
        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        
        # Extract network weights
        if 'network_weights' in checkpoint:
            state_dict = checkpoint['network_weights']
        else:
            state_dict = checkpoint
        
        # Get architecture config from checkpoint
        init_args = checkpoint.get('init_args', {})
        plans = init_args.get('plans', {})
        configuration = init_args.get('configuration', '2d')
        
        # Import PlainConvUNet from dynamic_network_architectures
        from dynamic_network_architectures.architectures.unet import PlainConvUNet
        import torch.nn as nn
        
        # Get architecture config from plans
        arch_config = plans['configurations'][configuration]['architecture']['arch_kwargs']
        
        # Build model with exact same architecture
        model = PlainConvUNet(
            input_channels=1,  # ACDC is single channel
            n_stages=arch_config['n_stages'],
            features_per_stage=arch_config['features_per_stage'],
            conv_op=nn.Conv2d,
            kernel_sizes=arch_config['kernel_sizes'],
            strides=arch_config['strides'],
            n_conv_per_stage=arch_config['n_conv_per_stage'],
            num_classes=NUM_CLASSES,
            n_conv_per_stage_decoder=arch_config['n_conv_per_stage_decoder'],
            conv_bias=arch_config['conv_bias'],
            norm_op=nn.InstanceNorm2d,
            norm_op_kwargs=arch_config['norm_op_kwargs'],
            dropout_op=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs=arch_config['nonlin_kwargs'],
            deep_supervision=False,
        ).to(DEVICE)
        
        model.load_state_dict(state_dict)
        model.eval()
        print(f"  ✓ Loaded nnUNet with PlainConvUNet architecture")
        return model
        
    except Exception as e:
        print(f"  nnUNet loading via dynamic_network_architectures failed: {e}")
        print(f"  Skipping nnUNet model")
        return None


def load_unetpp(weights_path):
    """Load UNet++ (NestedUNet) model."""
    unetpp_path = PROJECT_ROOT / 'comparison' / 'pytorch-nested-unet'
    if not unetpp_path.exists(): return None
    sys.path.insert(0, str(unetpp_path))
    
    try:
        from archs import NestedUNet
        
        # Create model - trained with 5 input channels (2.5D)
        model = NestedUNet(
            num_classes=NUM_CLASSES,
            input_channels=5,
            deep_supervision=False
        ).to(DEVICE)
        
        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        print(f"  ! Error loading UNet++: {e}")
        return None


def load_model(model_name, config):
    """Load model by type."""
    print(f"Loading {model_name}...")
    weights_path = config['weights']
    
    if not weights_path.exists():
        print(f"  WARNING: Weights not found: {weights_path}")
        return None
    
    if config['type'] == 'pie_unet':
        return load_pie_unet(weights_path)
    elif config['type'] == 'swin_unet':
        return load_swin_unet(weights_path)
    elif config['type'] == 'transunet':
        return load_transunet(weights_path)
    elif config['type'] == 'swinunetr':
        return load_swinunetr(weights_path)
    elif config['type'] == 'nnunet':
        return load_nnunet(weights_path)
    elif config['type'] == 'unetpp':
        return load_unetpp(weights_path)
    else:
        raise ValueError(f"Unknown model type: {config['type']}")


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================

def inference_pie_unet(model, image):
    """Inference with PIE-UNet (expects 5-channel 2.5D input)."""
    # If image is 2D (H, W), stack to simulate 2.5D
    if image.ndim == 2:
        image = np.stack([image] * 5, axis=0)  # (5, H, W)
    elif image.ndim == 3 and image.shape[0] != 5:
        # If passed tensor isn't 5 channel, take middle and stack?
        # Assuming input is correct 2.5D 5-slice block
        pass
    
    x = torch.from_numpy(image).unsqueeze(0).float().to(DEVICE)  # (1, 5, H, W)
    
    with torch.no_grad():
        outputs, _ = model(x)
        out = outputs[-1]  # Take last deep supervision output
        pred = torch.argmax(F.softmax(out, dim=1), dim=1).squeeze(0)
    
    return pred.cpu().numpy()


def inference_swin_unet(model, image):
    """Inference with Swin-Unet (expects 3 channel input)."""
    if image.ndim == 3:  # Take middle slice if 2.5D
        image = image[image.shape[0] // 2]
    
    h, w = image.shape
    if h != IMG_SIZE or w != IMG_SIZE:
        image_resized = zoom(image, (IMG_SIZE / h, IMG_SIZE / w), order=3)
    else:
        image_resized = image
    
    # Repeat grayscale to 3 channels (model trained with 3 channels)
    image_3ch = np.stack([image_resized, image_resized, image_resized], axis=0)  # (3, H, W)
    x_tensor = torch.from_numpy(image_3ch).unsqueeze(0).float().to(DEVICE)  # (1, 3, H, W)
    
    with torch.no_grad():
        out = model(x_tensor)
        pred = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
    
    pred_np = pred.cpu().numpy()
    
    if h != IMG_SIZE or w != IMG_SIZE:
        pred_np = zoom(pred_np, (h / IMG_SIZE, w / IMG_SIZE), order=0)
    
    return pred_np


def inference_transunet(model, image):
    """Inference with TransUNet (expects single channel input)."""
    if image.ndim == 3:  # Take middle slice if 2.5D
        image = image[image.shape[0] // 2]
    
    x, y = image.shape
    if x != IMG_SIZE or y != IMG_SIZE:
        image_resized = zoom(image, (IMG_SIZE / x, IMG_SIZE / y), order=3)
    else:
        image_resized = image
    
    x_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    
    with torch.no_grad():
        out = model(x_tensor)
        pred = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
    
    pred_np = pred.cpu().numpy()
    
    if x != IMG_SIZE or y != IMG_SIZE:
        pred_np = zoom(pred_np, (x / IMG_SIZE, y / IMG_SIZE), order=0)
    
    return pred_np


def inference_swinunetr(model, image):
    """Inference with SwinUNETR (expects single channel input)."""
    if image.ndim == 3:  # Take middle slice if 2.5D
        image = image[image.shape[0] // 2]
    
    h, w = image.shape
    if h != IMG_SIZE or w != IMG_SIZE:
        image_resized = zoom(image, (IMG_SIZE / h, IMG_SIZE / w), order=3)
    else:
        image_resized = image
    
    # Normalize to [0, 1]
    if image_resized.max() > image_resized.min():
        image_resized = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min())
    
    x_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    
    with torch.no_grad():
        out = model(x_tensor)
        pred = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
    
    pred_np = pred.cpu().numpy()
    
    if h != IMG_SIZE or w != IMG_SIZE:
        pred_np = zoom(pred_np, (h / IMG_SIZE, w / IMG_SIZE), order=0)
    
    return pred_np


def inference_nnunet(model, image):
    """Inference with nnUNet (expects single channel input with Z-score normalization)."""
    if image.ndim == 3:  # Take middle slice if 2.5D
        image = image[image.shape[0] // 2]
    
    h, w = image.shape
    
    # nnUNet uses Z-score normalization
    image_norm = image.astype(np.float32)
    mean = image_norm.mean()
    std = image_norm.std()
    if std > 0:
        image_norm = (image_norm - mean) / std
    
    # nnUNet 2D was trained with patch size [256, 224], resize to match
    nnunet_h, nnunet_w = 256, 224
    if h != nnunet_h or w != nnunet_w:
        image_resized = zoom(image_norm, (nnunet_h / h, nnunet_w / w), order=3)
    else:
        image_resized = image_norm
    
    x_tensor = torch.from_numpy(image_resized).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    
    with torch.no_grad():
        out = model(x_tensor)
        if isinstance(out, (list, tuple)):
            out = out[0]
        pred = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
    
    pred_np = pred.cpu().numpy()
    
    # Resize back to original size
    if h != nnunet_h or w != nnunet_w:
        pred_np = zoom(pred_np, (h / nnunet_h, w / nnunet_w), order=0)
    
    return pred_np


def inference_unetpp(model, image):
    """Inference with UNet++ (expects 5 channel 2.5D input)."""
    # UNet++ was trained with 5 input channels (2.5D)
    if image.ndim == 2:
        # Stack single slice to 5 channels
        image = np.stack([image] * 5, axis=0)
    elif image.ndim == 3 and image.shape[0] != 5:
        # e.g. 3 slices? stack? typically 2.5D uses neighbors.
        # Assuming we just replicate if not 5
        center = image[image.shape[0] // 2]
        image = np.stack([center] * 5, axis=0)
        
    # image should be (5, H, W)
    if image.ndim == 3:
        n, h, w = image.shape
    else:
        h, w = image.shape
        n = 1
    
    if h != IMG_SIZE or w != IMG_SIZE:
        # Resize each channel
        image_resized = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=image.dtype)
        for i in range(n):
            image_resized[i] = zoom(image[i], (IMG_SIZE / h, IMG_SIZE / w), order=3)
    else:
        image_resized = image
    
    # Normalize
    if image_resized.max() > image_resized.min():
        image_resized = (image_resized - image_resized.min()) / (image_resized.max() - image_resized.min())
    
    x_tensor = torch.from_numpy(image_resized).unsqueeze(0).float().to(DEVICE)  # (1, 5, H, W)
    
    with torch.no_grad():
        out = model(x_tensor)
        if isinstance(out, list):
            out = out[-1]  # Take last output if deep supervision
        pred = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
    
    pred_np = pred.cpu().numpy()
    
    if h != IMG_SIZE or w != IMG_SIZE:
        pred_np = zoom(pred_np, (h / IMG_SIZE, w / IMG_SIZE), order=0)
    
    return pred_np


def inference(model, image, model_type):
    """Run inference based on model type."""
    if model_type == 'pie_unet':
        return inference_pie_unet(model, image)
    elif model_type == 'swin_unet':
        return inference_swin_unet(model, image)
    elif model_type == 'transunet':
        return inference_transunet(model, image)
    elif model_type == 'swinunetr':
        return inference_swinunetr(model, image)
    elif model_type == 'nnunet':
        return inference_nnunet(model, image)
    elif model_type == 'unetpp':
        return inference_unetpp(model, image)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_single_model_figure(image, gt_mask, prediction, model_name, case_name, output_path):
    """Create figure showing only the model prediction overlay."""
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Get display image (middle slice for 2.5D)
    if image.ndim == 3:
        display_img = image[image.shape[0] // 2]
    else:
        display_img = image
    
    # Create custom colormap for segmentation
    colors_list = ['black', 'red', 'green', 'blue']  # BG, RV, MYO, LV
    cmap = mcolors.ListedColormap(colors_list)
    
    # Model prediction overlay on MRI (no title)
    ax.imshow(display_img, cmap='gray')
    pred_masked = np.ma.masked_where(prediction == 0, prediction)
    ax.imshow(pred_masked, cmap=cmap, alpha=0.6, vmin=0, vmax=NUM_CLASSES-1)
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()


def create_comparison_figure(image, gt_mask, predictions, model_names, case_name, output_path):
    """Create comparison figure using masked overlay technique (all models in one image)."""
    
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models + 2, figsize=(4 * (n_models + 2), 4))
    
    # Get display image (middle slice for 2.5D)
    if image.ndim == 3:
        display_img = image[image.shape[0] // 2]
    else:
        display_img = image
    
    # Create custom colormap for segmentation
    colors_list = ['black', 'red', 'green', 'blue']  # BG, RV, MYO, LV
    cmap = mcolors.ListedColormap(colors_list)
    
    # 1. Input image (original MRI)
    axes[0].imshow(display_img, cmap='gray')
    axes[0].set_title('Input MRI', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Ground Truth overlay (mask background)
    axes[1].imshow(display_img, cmap='gray')
    gt_masked = np.ma.masked_where(gt_mask == 0, gt_mask)  # Hide background
    axes[1].imshow(gt_masked, cmap=cmap, alpha=0.6, vmin=0, vmax=NUM_CLASSES-1)
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Model predictions overlay
    for i, (model_name, pred) in enumerate(zip(model_names, predictions)):
        axes[i + 2].imshow(display_img, cmap='gray')
        pred_masked = np.ma.masked_where(pred == 0, pred)  # Hide background
        axes[i + 2].imshow(pred_masked, cmap=cmap, alpha=0.6, vmin=0, vmax=NUM_CLASSES-1)
        axes[i + 2].set_title(model_name, fontsize=12, fontweight='bold')
        axes[i + 2].axis('off')
    
    # Add legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, color='red', label='Right Ventricle (RV)'),
        Rectangle((0, 0), 1, 1, color='green', label='Myocardium (MYO)'),
        Rectangle((0, 0), 1, 1, color='blue', label='Left Ventricle (LV)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)

    fig.suptitle(f'{case_name}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=(0, 0.08, 1, 0.95))
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def save_gt_only_figure(image, gt_mask, case_name, output_path):
    """Save ground truth only figure."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Get display image
    if image.ndim == 3:
        display_img = image[image.shape[0] // 2]
    else:
        display_img = image
    
    colors_list = ['black', 'red', 'green', 'blue']
    cmap = mcolors.ListedColormap(colors_list)
    
    # Ground Truth overlay (no title)
    ax.imshow(display_img, cmap='gray')
    gt_masked = np.ma.masked_where(gt_mask == 0, gt_mask)
    ax.imshow(gt_masked, cmap=cmap, alpha=0.6, vmin=0, vmax=NUM_CLASSES-1)
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()


def save_input_only_figure(image, case_name, output_path):
    """Save input MRI only figure."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Get display image
    if image.ndim == 3:
        display_img = image[image.shape[0] // 2]
    else:
        display_img = image
    
    # Input MRI (no title)
    ax.imshow(display_img, cmap='gray')
    ax.axis('off')
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_test_data(num_cases=-1):
    """Load test cases from ACDC preprocessed data.
    
    Args:
        num_cases: Number of cases to load. -1 means all slices.
    """
    try:
        from src.data.acdc_dataset import ACDCDataset2D
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        print("! Dataset imports failed")
        return []
    
    test_npy_dir = str(PROJECT_ROOT / 'preprocessed_data' / 'ACDC' / 'testing')
    
    print(f"Loading data from: {test_npy_dir}")
    # Using existing ACDCDataset2D class
    try:
        # Note: ACDCDataset2D loads ALL data in the dir. 
        # We will filter by patient later.
        test_dataset = ACDCDataset2D(
            npy_dir=test_npy_dir,
            use_memmap=True
        )
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")
        return []
    
    total_slices = len(test_dataset)
    print(f"Total available slices in directory: {total_slices}")
    
    cases = []
    
    # We iterate through all dataset items and select what we need
    # This might be slow if dataset is huge, but for ACDC (~1900 slices) it's fine.
    
    count = 0
    
    # If filtered patients, find their indices first
    indices_to_process = list(range(total_slices))
    
    for idx in indices_to_process:
        if num_cases != -1 and count >= num_cases:
            break
            
        vol_idx, slice_idx = test_dataset.index_map[idx]
        vol_path = test_dataset.vol_paths[vol_idx]
        vol_id = os.path.basename(vol_path).replace('.npy', '')
        # vol_id example: patient101_ED
        
        patient_id = vol_id.split('_')[0]
        
        if PATIENT_FILTER is not None and patient_id not in PATIENT_FILTER:
            continue
            
        # Load data
        image, mask = test_dataset[idx] # Returns numpy arrays usually for this class? 
        # ACDCDataset2D returns (img, target). img is (C, H, W) or (H,W,C)? 
        # Checking acdc_dataset.py: it returns tensor or numpy? 
        # It calls transforms if any. Default __getitem__ returns torch tensor if transformed.
        # But here valid Dataset returns? 
        # Let's assume it returns standard format.
        
        if isinstance(image, torch.Tensor):
            image_np = image.numpy()
        else:
            image_np = image
            
        if isinstance(mask, torch.Tensor):
            mask_np = mask.numpy()
        else:
            mask_np = mask
            
        # Ensure image is 2D or 3D properly
        # For visualization, if image is (C, H, W) and C=3 (RGB) or C=1 (Gray)
        if image_np.ndim == 3 and image_np.shape[0] in [1, 3]:
            # It's (C, H, W), take first channel for gray display
            image_np = image_np[0] 
        
        slice_name = f"{vol_id}_slice{slice_idx:03d}"
        
        cases.append({
            'image': image_np, # 2D array (H, W)
            'mask': mask_np,   # 2D array (H, W)
            'case_name': slice_name,
            'patient_id': patient_id,
            'vol_id': vol_id,
            'slice_idx': slice_idx
        })
        count += 1
    
    return cases


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("MODEL COMPARISON VISUALIZATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Models: {list(MODEL_CONFIGS.keys())}")
    print(f"Cases: {NUM_CASES}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    models = {}
    for model_name, config in MODEL_CONFIGS.items():
        try:
            model = load_model(model_name, config)
            if model is not None:
                models[model_name] = {'model': model, 'type': config['type']}
                print(f"  ✓ Loaded {model_name}")
        except Exception as e:
            print(f"  ✗ Failed to load {model_name}: {e}")
            # print traceback
            import traceback
            traceback.print_exc()
    
    if not models:
        print("No models loaded! Exiting.")
        return
    
    # Load test data
    print("\nLoading test data...")
    cases = load_test_data(NUM_CASES)
    print(f"Loaded {len(cases)} cases")
    
    if not cases:
        print("No cases found. Check data path and filters.")
        return
    
    # Generate visualizations - organized by patient folder
    print("\nGenerating visualizations...")
    print("Output structure: patient_folder/model_name/slice.png")
    
    for i, case_data in enumerate(cases):
        image = case_data['image']
        gt_mask = case_data['mask']
        case_name = case_data['case_name']
        patient_id = case_data['patient_id']
        vol_id = case_data['vol_id']
        slice_idx = case_data['slice_idx']
        
        print(f"Case {i+1}/{len(cases)}: {case_name}")
        
        # Create patient folder
        patient_dir = run_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate prediction for each model and save separately
        for model_name, model_info in models.items():
            # Create model subfolder within patient folder
            model_dir = patient_dir / model_name.replace(' ', '_').replace('+', 'p')
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Run inference
            pred = inference(model_info['model'], image, model_info['type'])
            
            # Save individual model visualization
            output_path = model_dir / f"{vol_id}_slice{slice_idx:03d}.png"
            create_single_model_figure(image, gt_mask, pred, model_name, case_name, output_path)
        
        # Save Ground Truth image in patient folder
        gt_dir = patient_dir / "GroundTruth"
        gt_dir.mkdir(parents=True, exist_ok=True)
        gt_output_path = gt_dir / f"{vol_id}_slice{slice_idx:03d}.png"
        save_gt_only_figure(image, gt_mask, case_name, gt_output_path)
        
        # Save Input MRI image in patient folder
        input_dir = patient_dir / "InputMRI"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_output_path = input_dir / f"{vol_id}_slice{slice_idx:03d}.png"
        save_input_only_figure(image, case_name, input_output_path)
    
    print("\n" + "=" * 70)
    print(f"VISUALIZATION COMPLETE!")
    print(f"Output directory: {run_dir}")
    print(f"Structure: {run_dir}/patient_id/model_name/slice.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
