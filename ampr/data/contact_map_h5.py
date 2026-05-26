"""Unified read-only access across train/valid/test contact-map HDF5 files."""

from pathlib import Path
from typing import Mapping

import h5py
import numpy as np


class ContactMapStore:
    """Opens 1-3 HDF5 files lazily; routes lookup to whichever has the key."""

    def __init__(self, paths: Mapping[str, Path]):
        self._handles = {k: h5py.File(str(v), "r") for k, v in paths.items()
                         if Path(v).exists()}

    def __getitem__(self, prot_id: str) -> np.ndarray:
        for h in self._handles.values():
            if prot_id in h:
                return h[prot_id][...].astype(np.float32)
        raise KeyError(prot_id)

    def __contains__(self, prot_id: str) -> bool:
        return any(prot_id in h for h in self._handles.values())

    def get_sequence(self, prot_id: str) -> str:
        for h in self._handles.values():
            if prot_id in h:
                return h[prot_id].attrs.get("sequence", "")
        raise KeyError(prot_id)

    def __del__(self):
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
