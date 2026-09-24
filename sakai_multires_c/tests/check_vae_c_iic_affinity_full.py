"""Independent, read-only NumPy spot check of a full IIC affinity payload."""

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np


ROWS = 8583
COLS = 180
H_SHA256 = "1807ea44b7e3c884b04f7a35195731806916374caf8297e0294e7aaa55abeb44"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h", required=True, type=Path)
    parser.add_argument("--affinity", required=True, type=Path)
    args = parser.parse_args()

    if args.h.stat().st_size != ROWS * COLS * 4:
        raise SystemExit("H byte count mismatch")
    if file_sha256(args.h) != H_SHA256:
        raise SystemExit("H source SHA-256 mismatch")
    if args.affinity.stat().st_size != ROWS * ROWS * 8:
        raise SystemExit("affinity byte count mismatch")

    h = np.memmap(args.h, dtype="<f4", mode="r", shape=(ROWS, COLS))
    affinity = np.memmap(args.affinity, dtype="<f8", mode="r", shape=(ROWS, ROWS))
    pairs = {
        (0, 0), (0, 1), (1, 0), (0, ROWS - 1),
        (ROWS - 1, 0), (ROWS - 1, ROWS - 1),
        (ROWS - 2, ROWS - 1), (ROWS // 2, ROWS // 2),
    }
    rng = np.random.default_rng(20260917)
    while len(pairs) < 2056:
        i, j = rng.integers(0, ROWS, size=2)
        pairs.add((int(i), int(j)))

    max_abs = -1.0
    max_rel = -1.0
    worst = None
    for i, j in sorted(pairs):
        left = h[i].astype(np.float64)
        right = h[j].astype(np.float64)
        diff = left - right
        expected = math.exp(-math.sqrt(float(np.sum(diff * diff, dtype=np.float64))))
        observed = float(affinity[i, j])
        abs_error = abs(expected - observed)
        rel_error = abs_error / max(abs(expected), np.finfo(np.float64).tiny)
        if abs_error > max_abs:
            max_abs = abs_error
            worst = (i, j, expected, observed)
        max_rel = max(max_rel, rel_error)
        if not math.isfinite(observed) or not math.isclose(
            expected, observed, rel_tol=5e-14, abs_tol=5e-15
        ):
            raise SystemExit(
                f"independent affinity mismatch ({i},{j}): "
                f"expected={expected:.17g} artifact={observed:.17g} "
                f"abs={abs_error:.17g} rel={rel_error:.17g}"
            )
    print(f"SAMPLE_COUNT={len(pairs)}")
    print(f"MAX_ABS={max_abs:.17g}")
    print(f"MAX_REL={max_rel:.17g}")
    print(f"WORST_PAIR={worst[0]},{worst[1]}")
    print(f"WORST_PYTHON={worst[2]:.17g}")
    print(f"WORST_ARTIFACT={worst[3]:.17g}")
    print("REFERENCE_SAMPLE_PASS=true")


if __name__ == "__main__":
    main()
