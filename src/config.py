"""
Configuration System for EGM-Net.

Loads YAML configuration and provides a dataclass-based interface.
Supports presets (baseline, lite, sota) for quick configuration.
"""

import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class BackboneConfig:
    """Backbone configuration."""
    type: str = "hrnet_mamba"  # "hrnet" or "hrnet_mamba"
    base_channels: int = 64
    num_stages: int = 4
    blocks_per_stage: int = 2
    low_res_multiplier: int = 2


@dataclass
class ComponentsConfig:
    """Auxiliary components toggle."""
    use_mamba: bool = True
    mamba_depth: int = 2
    mamba_scan_dim: int = 64
    use_spectral: bool = True
    spectral_threshold: float = 0.1
    use_dual_stream: bool = True


@dataclass
class CoarseHeadConfig:
    """Coarse head configuration."""
    type: str = "constellation"  # "linear" or "constellation"
    embedding_dim: int = 2
    init_gamma: float = 1.0


@dataclass
class FineHeadConfig:
    """Fine head configuration."""
    type: str = "implicit"  # "none" or "implicit"
    enabled: bool = True
    hidden_dim: int = 256
    num_layers: int = 4
    num_frequencies: int = 64
    use_energy_gating: bool = True


@dataclass
class FusionConfig:
    """Fusion strategy configuration."""
    type: str = "energy_gated"  # "simple" or "energy_gated"
    temperature: float = 1.0


@dataclass 
class HeadsConfig:
    """All heads configuration."""
    coarse: CoarseHeadConfig = field(default_factory=CoarseHeadConfig)
    fine: FineHeadConfig = field(default_factory=FineHeadConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)


@dataclass
class EyeOpeningLossConfig:
    """Eye opening loss configuration."""
    enabled: bool = True
    weight: float = 0.05
    warmup_epochs: int = 10
    radius: int = 5


@dataclass
class ConsistencyLossConfig:
    """Consistency loss configuration."""
    enabled: bool = True
    weight: float = 0.1


@dataclass
class LossConfig:
    """Loss functions configuration."""
    dice_weight: float = 1.0
    ce_weight: float = 0.5
    focal_weight: float = 0.0
    focal_gamma: float = 2.0
    boundary_weight: float = 0.1
    spectral_weight: float = 0.0
    eye_opening: EyeOpeningLossConfig = field(default_factory=EyeOpeningLossConfig)
    consistency: ConsistencyLossConfig = field(default_factory=ConsistencyLossConfig)


@dataclass
class PointSamplingConfig:
    """Point sampling configuration."""
    enabled: bool = True
    num_samples: int = 4096
    strategy: str = "uncertainty_energy"  # "random", "uncertainty", "energy", "uncertainty_energy"


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    random_flip: bool = True
    random_rotation: bool = True
    rotation_range: int = 15


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    point_sampling: PointSamplingConfig = field(default_factory=PointSamplingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class InferenceConfig:
    """Inference configuration."""
    use_implicit: bool = True
    output_scale: float = 1.0
    tta_enabled: bool = False


@dataclass
class ModelConfig:
    """Main model configuration."""
    name: str = "EGMNet"
    in_channels: int = 3
    num_classes: int = 4
    img_size: int = 256
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    components: ComponentsConfig = field(default_factory=ComponentsConfig)


@dataclass
class EGMNetConfig:
    """Root configuration for EGM-Net."""
    model: ModelConfig = field(default_factory=ModelConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    preset: str = "sota"


# =============================================================================
# Preset Configurations
# =============================================================================

PRESETS = {
    "baseline": {
        "model": {
            "backbone": {"type": "hrnet"},
            "components": {
                "use_mamba": False,
                "use_spectral": False,
                "use_dual_stream": True
            }
        },
        "heads": {
            "coarse": {"type": "linear"},
            "fine": {"enabled": False}
        }
    },
    "lite": {
        "model": {
            "backbone": {"type": "hrnet", "base_channels": 32, "num_stages": 3},
            "components": {
                "use_mamba": False,
                "use_spectral": False,
                "use_dual_stream": True
            }
        },
        "heads": {
            "coarse": {"type": "constellation"},
            "fine": {"enabled": False}
        }
    },
    "sota": {
        "model": {
            "backbone": {"type": "hrnet_mamba"},
            "components": {
                "use_mamba": True,
                "use_spectral": True,
                "use_dual_stream": True
            }
        },
        "heads": {
            "coarse": {"type": "constellation"},
            "fine": {"enabled": True, "use_energy_gating": True}
        }
    }
}


def deep_update(base: dict, update: dict) -> dict:
    """Recursively update nested dictionaries."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def dict_to_dataclass(data: dict, cls):
    """Convert nested dictionary to dataclass instances."""
    if not hasattr(cls, '__dataclass_fields__'):
        return data
    
    kwargs = {}
    for field_name, field_info in cls.__dataclass_fields__.items():
        if field_name in data:
            value = data[field_name]
            field_type = field_info.type
            
            # Handle nested dataclasses
            if hasattr(field_type, '__dataclass_fields__'):
                kwargs[field_name] = dict_to_dataclass(value, field_type)
            else:
                kwargs[field_name] = value
    
    return cls(**kwargs)


def load_config(config_path: Optional[str] = None) -> EGMNetConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, uses defaults.
        
    Returns:
        EGMNetConfig dataclass instance.
    """
    # Start with default config as dict
    import dataclasses
    
    def dataclass_to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: dataclass_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        return obj
    
    config_dict = dataclass_to_dict(EGMNetConfig())
    
    # Load from YAML if provided
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Apply preset first if specified
            preset = yaml_config.get('preset', 'sota')
            if preset in PRESETS:
                deep_update(config_dict, PRESETS[preset])
            
            # Then apply user overrides
            deep_update(config_dict, yaml_config)
    
    # Convert to dataclass
    return dict_to_dataclass(config_dict, EGMNetConfig)


def get_model_kwargs(config: EGMNetConfig) -> dict:
    """
    Extract model initialization kwargs from config.
    
    Returns:
        Dictionary of kwargs for EGMNet constructor.
    """
    return {
        'in_channels': config.model.in_channels,
        'num_classes': config.model.num_classes,
        'img_size': config.model.img_size,
        'base_channels': config.model.backbone.base_channels,
        'num_stages': config.model.backbone.num_stages,
        'use_hrnet': config.model.backbone.type == "hrnet_mamba",
        'use_mamba': config.model.components.use_mamba,
        'use_spectral': config.model.components.use_spectral,
        'implicit_hidden': config.heads.fine.hidden_dim,
        'implicit_layers': config.heads.fine.num_layers,
        'num_frequencies': config.heads.fine.num_frequencies,
        'use_fine_head': config.heads.fine.enabled,
        'coarse_head_type': config.heads.coarse.type,
        'fusion_type': config.heads.fusion.type,
    }


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test loading default config
    print("=== Default Config ===")
    config = load_config()
    print(f"Preset: {config.preset}")
    print(f"Backbone: {config.model.backbone.type}")
    print(f"Use Mamba: {config.model.components.use_mamba}")
    print(f"Use Spectral: {config.model.components.use_spectral}")
    print(f"Fine Head Enabled: {config.heads.fine.enabled}")
    
    # Test loading from file
    print("\n=== From config.yaml ===")
    config = load_config("../config.yaml")
    print(f"Preset: {config.preset}")
    print(f"Model: {get_model_kwargs(config)}")
