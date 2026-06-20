"""Test the init-from checkpoint loader helper."""
import torch
import torch.nn as nn

from ampr.training.trainer import load_checkpoint_weights


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


def test_loads_model_state(tmp_path):
    src = Tiny()
    with torch.no_grad():
        src.fc.weight.fill_(1.234)
    ckpt = tmp_path / "best.pt"
    torch.save({"epoch": 7, "model": src.state_dict()}, ckpt)

    dst = Tiny()
    info = load_checkpoint_weights(dst, str(ckpt), map_location="cpu")

    assert torch.allclose(dst.fc.weight, torch.full_like(dst.fc.weight, 1.234))
    assert info["epoch"] == 7
    assert info["missing"] == [] and info["unexpected"] == []


def test_accepts_bare_state_dict(tmp_path):
    src = Tiny()
    ckpt = tmp_path / "bare.pt"
    torch.save(src.state_dict(), ckpt)
    dst = Tiny()
    info = load_checkpoint_weights(dst, str(ckpt), map_location="cpu")
    assert info["epoch"] is None
