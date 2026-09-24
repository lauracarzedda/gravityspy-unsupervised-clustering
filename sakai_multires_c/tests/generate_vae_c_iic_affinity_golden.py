"""Independent NumPy golden for tiny synthetic and canonical-H IIC affinity.

Usage: python generate_vae_c_iic_affinity_golden.py TEST_EXECUTABLE H_DIRECTORY
Prints comparisons only; creates no files and does not read labels.
"""

from __future__ import annotations

import hashlib
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np


SELECTED_ROWS = (0, 1, 2, 100, 8582)
H_ROWS = 8583
H_COLS = 180
EXPECTED_H_SHA256 = "1807ea44b7e3c884b04f7a35195731806916374caf8297e0294e7aaa55abeb44"
EXPECTED_ROWS_SHA256 = "6e09d72e3b54f213a2096edffca627fd35f1f0e35778539f05a810a788d7c642"
ATOL = 5.0e-15
RTOL = 5.0e-14


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_fresh(executable: Path, *args: str) -> bytes:
    return subprocess.check_output([str(executable), *args], stderr=subprocess.PIPE)


def parse_dump(payload: bytes, n: int, d: int | None = None) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    a = np.full((n, n), np.nan, dtype=np.float64)
    features: dict[tuple[int, int], float] = {}
    affinity_count = 0
    for raw_line in payload.decode("ascii").splitlines():
        kind, row_text, column_text, value_text = raw_line.split(",", 3)
        row, column = int(row_text), int(column_text)
        if kind == "A":
            if not (0 <= row < n and 0 <= column < n) or np.isfinite(a[row, column]):
                raise ValueError("duplicate or invalid affinity index")
            a[row, column] = float(value_text)
            affinity_count += 1
        elif kind == "H" and d is not None:
            if row not in SELECTED_ROWS or not 0 <= column < d or (row, column) in features:
                raise ValueError("duplicate or invalid feature index")
            features[(row, column)] = float.fromhex(value_text)
        else:
            raise ValueError(f"unexpected dump record {kind}")
    if affinity_count != n * n or not np.isfinite(a).all():
        raise ValueError("incomplete affinity dump")
    if d is not None and len(features) != n * d:
        raise ValueError("incomplete H feature dump")
    return a, features


def reference(h: np.ndarray) -> np.ndarray:
    h64 = h.astype(np.float64)
    n = h64.shape[0]
    a = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        a[i, i] = 1.0
        for j in range(i + 1, n):
            diff = h64[i] - h64[j]
            distance = np.sqrt(np.sum(diff * diff, dtype=np.float64))
            value = np.exp(-distance)
            a[i, j] = value
            a[j, i] = value
    return a


def compare(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.shape != expected.shape:
        raise ValueError(f"{name}: shape mismatch")
    if not (np.isfinite(actual).all() and (actual > 0.0).all() and (actual <= 1.0).all()):
        raise ValueError(f"{name}: invalid affinity range")
    if not np.all(actual.diagonal() == 1.0):
        raise ValueError(f"{name}: diagonal is not exactly one")
    if not np.array_equal(actual, actual.T):
        raise ValueError(f"{name}: symmetry is not bitwise exact")
    error = np.abs(actual - expected)
    relative = error / np.abs(expected)
    worst = np.unravel_index(int(np.argmax(error)), error.shape)
    print(f"{name}_COUNT={actual.size}")
    print(f"{name}_MAX_ABS={float(np.max(error)):.17g}")
    print(f"{name}_MAX_REL={float(np.max(relative)):.17g}")
    print(f"{name}_WORST_INDEX={worst[0]},{worst[1]}")
    print(f"{name}_WORST_PYTHON={expected[worst]:.17g}")
    print(f"{name}_WORST_C={actual[worst]:.17g}")
    if not np.all(error <= ATOL + RTOL * np.abs(expected)):
        raise ValueError(f"{name}: C/NumPy tolerance exceeded")


def canonical_rows(path: Path) -> None:
    rows = path.read_text(encoding="utf-8").splitlines()
    if len(rows) != H_ROWS + 1:
        raise ValueError("canonical H row-map length mismatch")
    if rows[0] != "manifest_index,event_index,event_id,source_split,split_pos":
        raise ValueError("canonical H row-map header mismatch")
    for position in SELECTED_ROWS:
        fields = rows[position + 1].split(",")
        if len(fields) != 5 or int(fields[0]) != position or not fields[2]:
            raise ValueError(f"canonical row identity mismatch at {position}")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: script TEST_EXECUTABLE H_DIRECTORY")
    executable = Path(sys.argv[1])
    directory = Path(sys.argv[2])
    payload_path = directory / "h_f32.bin"
    rows_path = directory / "rows.csv"
    if sha256(payload_path) != EXPECTED_H_SHA256 or sha256(rows_path) != EXPECTED_ROWS_SHA256:
        raise ValueError("canonical H identity mismatch")
    canonical_rows(rows_path)

    synthetic = np.array([
        [0.125, -0.5, 1.25, 0.1],
        [-0.75, 0.25, 1.25, 0.375],
        [0.125, -0.5, 1.25, 0.1],
        [1.5, 0.2, -0.75, -0.625],
        [-1.125, 0.75, 0.5, 1.75],
    ], dtype=np.float32)
    synthetic_dump = run_fresh(executable, "--dump-synthetic")
    if synthetic_dump != run_fresh(executable, "--dump-synthetic"):
        raise ValueError("synthetic fresh-process output differs")
    synthetic_c, _ = parse_dump(synthetic_dump, len(synthetic))
    compare("SYNTHETIC", synthetic_c, reference(synthetic))

    data = payload_path.read_bytes()
    if len(data) != H_ROWS * H_COLS * struct.calcsize("<f"):
        raise ValueError("canonical H byte length mismatch")
    all_h = np.frombuffer(data, dtype="<f4").reshape(H_ROWS, H_COLS)
    selected = all_h[list(SELECTED_ROWS)]
    real_dump = run_fresh(executable, "--dump-real", str(directory))
    if real_dump != run_fresh(executable, "--dump-real", str(directory)):
        raise ValueError("real-H fresh-process output differs")
    real_c, c_features = parse_dump(real_dump, len(SELECTED_ROWS), H_COLS)
    for row_number, manifest_index in enumerate(SELECTED_ROWS):
        for feature in range(H_COLS):
            if c_features[(manifest_index, feature)] != float(selected[row_number, feature]):
                raise ValueError(f"feature identity mismatch at {manifest_index},{feature}")
    print(f"REAL_H_FEATURES_VERIFIED={len(c_features)}")
    compare("REAL_H_TINY", real_c, reference(selected))
    print(f"ATOL={ATOL:.17g} RTOL={RTOL:.17g}")
    print("IIC_AFFINITY_PYTHON_GOLDEN_PASS=true")


if __name__ == "__main__":
    main()
