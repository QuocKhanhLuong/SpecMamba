import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch  # noqa: F401
except ImportError:
    torch_stub = types.ModuleType("torch")
    nn_stub = types.ModuleType("torch.nn")
    functional_stub = types.ModuleType("torch.nn.functional")
    utils_stub = types.ModuleType("torch.utils")
    data_stub = types.ModuleType("torch.utils.data")

    class _Module:
        pass

    class _Dataset:
        pass

    nn_stub.Module = _Module
    data_stub.Dataset = _Dataset
    data_stub.DataLoader = object
    data_stub.Subset = object
    torch_stub.nn = nn_stub
    utils_stub.data = data_stub

    sys.modules["torch"] = torch_stub
    sys.modules["torch.nn"] = nn_stub
    sys.modules["torch.nn.functional"] = functional_stub
    sys.modules["torch.utils"] = utils_stub
    sys.modules["torch.utils.data"] = data_stub

    acdc_dataset_stub = types.ModuleType("data.acdc_dataset")
    acdc_dataset_stub.ACDCDataset2D = object
    acdc_dataset_stub.ACDCDataset2DAugmented = object
    sys.modules["data.acdc_dataset"] = acdc_dataset_stub

    loss_stub = types.ModuleType("losses.sota_loss")
    loss_stub.CombinedSOTALoss = object
    sys.modules["losses.sota_loss"] = loss_stub

from training import train_acdc


def test_patient_level_split_manifest_keeps_frames_together(tmp_path):
    volume_ids = [
        "patient001_ED",
        "patient001_ES",
        "patient002_ED",
        "patient002_ES",
        "patient003_ED",
        "patient003_ES",
        "patient004_ED",
        "patient004_ES",
    ]
    manifest_path = tmp_path / "splits" / "acdc_seed7.json"

    manifest = train_acdc.create_acdc_patient_split_manifest(
        volume_ids=volume_ids,
        seed=7,
        train_fraction=0.5,
        output_path=manifest_path,
    )

    assert manifest_path.exists()
    saved = json.loads(manifest_path.read_text())
    assert saved == manifest

    train_patients = set(manifest["splits"]["train"]["patients"])
    val_patients = set(manifest["splits"]["val"]["patients"])
    assert train_patients
    assert val_patients
    assert train_patients.isdisjoint(val_patients)

    for patient_id in train_patients:
        assert f"{patient_id}_ED" in manifest["splits"]["train"]["volumes"]
        assert f"{patient_id}_ES" in manifest["splits"]["train"]["volumes"]
    for patient_id in val_patients:
        assert f"{patient_id}_ED" in manifest["splits"]["val"]["volumes"]
        assert f"{patient_id}_ES" in manifest["splits"]["val"]["volumes"]


def test_indices_from_manifest_follow_dataset_volume_order(tmp_path):
    class DummyDataset:
        vol_paths = [
            str(tmp_path / "patient010_ES.npy"),
            str(tmp_path / "patient001_ED.npy"),
            str(tmp_path / "patient010_ED.npy"),
        ]
        index_map = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)]

    manifest = {
        "splits": {
            "train": {"volumes": ["patient010_ED", "patient010_ES"]},
            "val": {"volumes": ["patient001_ED"]},
        }
    }

    train_idx, val_idx = train_acdc.indices_from_acdc_split_manifest(
        DummyDataset(), manifest
    )

    assert train_idx == [0, 1, 3, 4, 5]
    assert val_idx == [2]


def test_indices_accept_flat_manifest_schema_from_split_script(tmp_path):
    class DummyDataset:
        vol_paths = [
            str(tmp_path / "patient010_ES.npy"),
            str(tmp_path / "patient001_ED.npy"),
            str(tmp_path / "patient010_ED.npy"),
        ]
        index_map = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)]

    manifest = {
        "split_type": "patient",
        "train_patients": ["patient010"],
        "val_patients": ["patient001"],
        "train_volumes": ["patient010_ED", "patient010_ES"],
        "val_volumes": ["patient001_ED"],
    }

    train_idx, val_idx = train_acdc.indices_from_acdc_split_manifest(
        DummyDataset(), manifest
    )

    assert train_idx == [0, 1, 3, 4, 5]
    assert val_idx == [2]


def test_load_training_config_flattens_sections(tmp_path):
    config_path = tmp_path / "acdc.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  data_dir: preprocessed_data/ACDC/training",
                "  batch_size: 2",
                "model:",
                "  model: specmamba",
                "  base_channels: 16",
                "training:",
                "  epochs: 3",
                "  seed: 123",
                "split:",
                "  train_fraction: 0.75",
                "  split_manifest: splits/acdc_seed123.json",
            ]
        )
    )

    config = train_acdc.load_training_config(config_path)

    assert config["data_dir"] == "preprocessed_data/ACDC/training"
    assert config["batch_size"] == 2
    assert config["base_channels"] == 16
    assert config["epochs"] == 3
    assert config["seed"] == 123
    assert config["train_fraction"] == 0.75
    assert config["split_manifest"] == "splits/acdc_seed123.json"


def test_parse_args_uses_config_defaults_and_cli_overrides(tmp_path):
    config_path = tmp_path / "acdc.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  batch_size: 2",
                "training:",
                "  epochs: 3",
                "  seed: 123",
                "output:",
                "  exp_name: from_config",
            ]
        )
    )

    args = train_acdc.parse_args(
        ["--config", str(config_path), "--epochs", "9", "--exp_name", "from_cli"]
    )

    assert args.batch_size == 2
    assert args.seed == 123
    assert args.epochs == 9
    assert args.exp_name == "from_cli"
