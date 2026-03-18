from __future__ import annotations
import json, sqlite3, uuid, datetime, zlib
from dataclasses import dataclass
from typing import Any, Optional, Dict, Tuple
import numpy as np

def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _pack_array(arr: np.ndarray) -> tuple[bytes, str, str]:
    arr = np.asarray(arr)
    raw = arr.tobytes(order="C")
    blob = zlib.compress(raw, level=6)
    shape = ",".join(map(str, arr.shape))
    dtype = str(arr.dtype)
    return blob, shape, dtype

def _unpack_array(blob: bytes, shape: str, dtype: str) -> np.ndarray:
    raw = zlib.decompress(blob)
    shp = tuple(int(x) for x in shape.split(",") if x)
    arr = np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shp)
    return arr

@dataclass
class RunMeta:
    model_name: Optional[str]
    explainer: str
    dataset_name: Optional[str] = None
    seed: Optional[int] = None
    params: Optional[Dict[str, Any]] = None

class SQLiteStore:
    def __init__(self, path: str = "mosaic_shap_runs.db"):
        self.path = path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA foreign_keys = ON;")
        return con

    def _init_db(self) -> None:
        con = self._connect()
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS runs (
          id TEXT PRIMARY KEY,
          created_at TEXT NOT NULL,
          model_name TEXT,
          explainer TEXT NOT NULL,
          dataset_name TEXT,
          seed INTEGER,
          params_json TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS artifacts (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id TEXT NOT NULL,
          name TEXT NOT NULL,
          array_shape TEXT NOT NULL,
          array_dtype TEXT NOT NULL,
          blob BLOB NOT NULL,
          FOREIGN KEY(run_id) REFERENCES runs(id)
        )
        """)
        con.commit()
        con.close()

    def create_run(self, meta: RunMeta) -> str:
        run_id = str(uuid.uuid4())
        con = self._connect()
        con.execute(
            "INSERT INTO runs(id, created_at, model_name, explainer, dataset_name, seed, params_json) VALUES (?,?,?,?,?,?,?)",
            (
                run_id,
                _now_iso(),
                meta.model_name,
                meta.explainer,
                meta.dataset_name,
                meta.seed,
                json.dumps(meta.params or {}),
            ),
        )
        con.commit()
        con.close()
        return run_id

    def save_array(self, run_id: str, name: str, arr: np.ndarray) -> None:
        blob, shape, dtype = _pack_array(arr)
        con = self._connect()
        con.execute(
            "INSERT INTO artifacts(run_id, name, array_shape, array_dtype, blob) VALUES (?,?,?,?,?)",
            (run_id, name, shape, dtype, blob),
        )
        con.commit()
        con.close()

    def load_array(self, run_id: str, name: str) -> np.ndarray:
        con = self._connect()
        row = con.execute(
            "SELECT array_shape, array_dtype, blob FROM artifacts WHERE run_id=? AND name=? ORDER BY id DESC LIMIT 1",
            (run_id, name),
        ).fetchone()
        con.close()
        if row is None:
            raise KeyError(f"No artifact '{name}' for run_id={run_id}")
        shape, dtype, blob = row
        return _unpack_array(blob, shape, dtype)
