"""Independent NumPy cross-check for common mini contact-sheet ranking.

This script never reads labels and never writes repository artifacts.  It
reassembles the frozen central-mu matrices directly from their binary files
and row maps, then compares the selected rows with representatives.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np


PROFILES = {
    "mini75_primary": {
        "rows": 8583,
        "latent": 32,
        "embedding": "mini75_experiment_v1_full_best_epoch47_posterior_mu_central_v1",
        "spectral": "mini75_iic_v1_737da555866b822e14c545c4f2687869f41a52cd1340b9e941f3a1ce77763eb9_spectral_primary_v1",
        "assignment_sha": "cc9e47a9d1a63c245bb53393c0fb65aa014015a95bbc395f3dc0f41554879ffd",
        "manifest_order": True,
    },
    "mini112_primary": {
        "rows": 8583,
        "latent": 64,
        "embedding": "mini112_primary_v1_full_best_epoch97_posterior_mu_central_v1",
        "spectral": "mini112_iic_v1_0cecf5cd656d0d61a237d50fcf96a7890fb2e6d052ef403051bc9bc17aed8fba_spectral_primary_v1",
        "assignment_sha": "17200b8d006962e39a7b49b07030042a0980970f57085b6318165cc30e738d3f",
        "manifest_order": True,
    },
    "mini224_1000_primary": {
        "rows": 1000,
        "latent": 64,
        "embedding": "mini224_bridge_v1_subset1000_best_epoch99_posterior_mu_central_v1",
        "spectral": "mini224_iic_v1_4236df3f9e82f0528fcea947b2e15e42fab8159c0d4b1ed03189e4a85daae8a5_spectral_primary_v1",
        "assignment_sha": "61bcc7a38b49c210626bd5fcf191f7b9171017df85fc7a1373fd674001af75d2",
        "manifest_order": False,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_central(repo: Path, profile: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    directory = repo / "sakai_multires_c/results/vae_c/embeddings" / str(profile["embedding"])
    latent = int(profile["latent"])
    matrices: list[np.ndarray] = []
    mappings: list[np.ndarray] = []
    for split in ("train", "validation"):
        matrix = np.fromfile(directory / f"{split}_mu_f32.bin", dtype="<f4")
        rows = load_rows(directory / f"{split}_rows.csv")
        matrix = matrix.reshape(len(rows), latent)
        matrices.append(matrix)
        mappings.append(np.asarray([int(row["manifest_index"]) for row in rows], dtype=np.int64))
    native = np.concatenate(matrices, axis=0)
    manifest_indices = np.concatenate(mappings, axis=0)
    if bool(profile["manifest_order"]):
        result = np.empty_like(native)
        result[manifest_indices] = native
        return result, np.arange(int(profile["rows"]), dtype=np.int64)
    return native, manifest_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--representatives", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repository_root.resolve(strict=True)
    profile = PROFILES[args.profile]
    spectral = repo / "sakai_multires_c/results/vae_c/iic" / str(profile["spectral"])
    assignment_path = spectral / "cluster_labels_i32.bin"
    assert sha256(assignment_path) == profile["assignment_sha"]
    assignments = np.fromfile(assignment_path, dtype="<i4")
    mu, manifest_indices = load_central(repo, profile)
    assert mu.shape == (int(profile["rows"]), int(profile["latent"]))
    assert assignments.shape == (int(profile["rows"]),)
    norms = np.linalg.norm(mu.astype(np.float64), axis=1, keepdims=True)
    assert np.all(np.isfinite(norms)) and np.all(norms > 0.0)
    unit = mu.astype(np.float64) / norms
    expected: list[tuple[int, int, int, int, float]] = []
    for cluster in range(36):
        members = np.flatnonzero(assignments == cluster)
        assert members.size >= 5
        prototype = unit[members].mean(axis=0)
        prototype /= np.linalg.norm(prototype)
        similarities = unit[members] @ prototype
        order = np.lexsort((manifest_indices[members], -similarities))
        for rank, offset in enumerate(order[:5], start=1):
            row = int(members[offset])
            expected.append((cluster, rank, row, int(manifest_indices[row]),
                             float(similarities[offset])))
    with args.representatives.open("r", encoding="utf-8", newline="") as stream:
        actual = list(csv.DictReader(stream))
    assert len(actual) == len(expected) == 180
    max_abs = 0.0
    for record, reference in zip(actual, expected, strict=True):
        cluster, rank, row, manifest_index, similarity = reference
        assert int(record["cluster_id"]) == cluster
        assert int(record["rank"]) == rank
        assert int(record["iic_row"]) == row
        assert int(record["manifest_index"]) == manifest_index
        max_abs = max(max_abs, abs(float(record["similarity"]) - similarity))
    assert max_abs <= 2e-15, max_abs
    print(f"PROFILE={args.profile}")
    print("REPRESENTATIVES=180")
    print(f"MAX_ABS_SIMILARITY={max_abs:.17g}")
    print("NUMPY_RANKING_CROSSCHECK_PASS=true")


if __name__ == "__main__":
    main()
