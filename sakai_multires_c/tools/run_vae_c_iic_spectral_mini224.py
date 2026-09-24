"""Label-free mini224_1000 bridge using the frozen mini112 spectral chain.

Technical full output is allowed only outside the repository.  Scientific
publication additionally requires this bridge to be committed and Git clean.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
import run_vae_c_iic_spectral_mini112 as reference  # noqa: E402

import numpy as np
import threadpoolctl


N = 1000
K = 36
RUN_ID = "mini224_iic_v1_4236df3f9e82f0528fcea947b2e15e42fab8159c0d4b1ed03189e4a85daae8a5"
AFFINITY_SHA = "813b37888d2e32bfc04525821ed5ce80915c5a5f99328da268bfb2ea4c1aab0a"
ROWS_SHA = "d7f5101cfe3f3ae31f24b12833f0125873888ff9ed33c4c33c428b90955db4c2"
SPEC_SHA = "b5b31fc19ca59a8f4f71e2986ceca804cf1bf24ae5422a3dee41df0124ce6a11"
CHECKSUM_SHA = "7f91b5a4e142f692967a41588e02215ef545efd2fd842344c2612d95734a2402"
SOURCE_H_SHA = "134e9a2efd12dfff57c3400fc3680134c83659c0f3314ced2a8468ed1941bea4"
SOURCE_H_SPEC_SHA = "01211e4507045320a6ae47bb868f1cf82e3a81930a964bed69a773e7b8647280"
IIC_FINAL_SHA = "e98dd0dd6513bc9636fe22af0341c41266bc00ce2629499e3aea2b5e50d83fe4"
REFERENCE_BRIDGE_SIZE = 26_484
REFERENCE_BRIDGE_SHA = "946e4817dae556b0e11efbcb5503c577296e760865a161709a1e7dab9e2d336d"
NOTEBOOK_SIZE = reference.NOTEBOOK_SIZE
NOTEBOOK_SHA = reference.NOTEBOOK_SHA
NOTEBOOK_HISTORICAL_SHA = reference.NOTEBOOK_HISTORICAL_SHA
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "sakai_multires_c" / "results"
AFFINITY_DIR = RESULTS_ROOT / "vae_c" / "iic" / (RUN_ID + "_affinity_primary_v1")
OFFICIAL_OUTPUT = RESULTS_ROOT / "vae_c" / "iic" / (RUN_ID + "_spectral_primary_v1")
ROW_HEADER = ["manifest_index", "event_index", "event_id", "source_split", "split_pos"]
MIN_AVAILABLE_RAM = 750_000_000
ESTIMATED_PEAK_RAM = 500_000_000
MIN_FREE_DISK = 32_000_000


def verify_reference() -> None:
    path = Path(reference.__file__).resolve()
    reference.verify_file(path, REFERENCE_BRIDGE_SIZE, REFERENCE_BRIDGE_SHA)
    if (reference.K, reference.NCV) != (K, 145):
        raise ValueError("frozen spectral chain configuration mismatch")


def validate_rows(path: Path) -> dict[str, int]:
    seen_manifest: set[int] = set()
    seen_events: set[str] = set()
    counts = {"train": 0, "validation": 0}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ROW_HEADER:
            raise ValueError("invalid label-free row-map header")
        for position, row in enumerate(reader):
            if set(row) != set(ROW_HEADER) or position >= N:
                raise ValueError("malformed or excess row-map record")
            try:
                manifest_index = int(row["manifest_index"])
                event_index = int(row["event_index"])
                split_pos = int(row["split_pos"])
            except (TypeError, ValueError) as error:
                raise ValueError("non-integer row identity") from error
            split = "train" if position < 800 else "validation"
            expected_pos = position if position < 800 else position - 800
            event_id = row["event_id"]
            if (manifest_index < 0 or manifest_index >= 8583
                    or manifest_index in seen_manifest or event_index != manifest_index + 1
                    or not event_id or event_id in seen_events
                    or row["source_split"] != split or split_pos != expected_pos):
                raise ValueError("mini224 row identity/order mismatch")
            seen_manifest.add(manifest_index)
            seen_events.add(event_id)
            counts[split] += 1
    if counts != {"train": 800, "validation": 200}:
        raise ValueError("mini224 row population mismatch")
    return counts


def validate_inputs(directory: Path, notebook_path: Path, full_scan: bool) -> np.memmap:
    directory = Path(directory)
    reference.verify_file(directory / "affinity_f64.bin", 8_000_000, AFFINITY_SHA)
    reference.verify_file(directory / "rows.csv", 34_607, ROWS_SHA)
    reference.verify_file(directory / "affinity_spec.txt", 1379, SPEC_SHA)
    reference.verify_file(directory / "sha256.txt", 242, CHECKSUM_SHA)
    reference.verify_file(notebook_path, NOTEBOOK_SIZE, NOTEBOOK_SHA)
    h_dir = directory.parent / (RUN_ID + "_h_main_v1")
    reference.verify_file(h_dir / "h_f32.bin", 720_000, SOURCE_H_SHA)
    reference.verify_file(h_dir / "h_spec.txt", 4393, SOURCE_H_SPEC_SHA)
    reference.verify_file(directory.parent / RUN_ID / "final.ckpt", 12_901_116, IIC_FINAL_SHA)
    if {path.name for path in directory.iterdir()} != {
        "affinity_f64.bin", "rows.csv", "affinity_spec.txt", "sha256.txt"
    }:
        raise ValueError("canonical affinity inventory mismatch")
    spec: dict[str, str] = {}
    for line in (directory / "affinity_spec.txt").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in spec:
            raise ValueError("malformed affinity spec")
        spec[key] = value
    required = {
        "format": "VAE_C_IIC_AFFINITY_ARTIFACT_V1", "source_mode": "canonical_mini224",
        "iic_run_id": RUN_ID, "rows": str(N), "cols": str(N),
        "dtype": "float64_ieee754", "endian": "little", "layout": "row_major",
        "payload_sha256": AFFINITY_SHA, "row_map_sha256": ROWS_SHA,
        "h_payload_sha256": SOURCE_H_SHA, "h_metadata_sha256": SOURCE_H_SPEC_SHA,
        "iic_final_sha256": IIC_FINAL_SHA, "diagonal": "exact_1.0",
        "symmetry": "upper_once_lower_exact_copy", "label_free": "true",
        "notebook06_reference_sha256": NOTEBOOK_HISTORICAL_SHA,
        "formula": "exp(-euclidean_l2(H_i,H_j))",
    }
    if any(spec.get(key) != value for key, value in required.items()):
        raise ValueError("affinity spec identity mismatch")
    for line in (directory / "sha256.txt").read_text(encoding="ascii").splitlines():
        sha, separator, name = line.partition("  ")
        if not separator or name not in {"affinity_f64.bin", "rows.csv", "affinity_spec.txt"}:
            raise ValueError("invalid affinity checksum manifest")
        if reference.sha256_file(directory / name) != sha:
            raise ValueError("affinity checksum manifest mismatch")
    validate_rows(directory / "rows.csv")
    array = np.memmap(directory / "affinity_f64.bin", dtype="<f8", mode="r", shape=(N, N))
    if full_scan:
        for start in range(0, N, 128):
            stop = min(N, start + 128)
            block = np.asarray(array[start:stop, :])
            if (not np.isfinite(block).all() or np.any(block <= 0.0)
                    or np.any(block > 1.0)
                    or not np.array_equal(np.diag(block[:, start:stop]), np.ones(stop - start))
                    or not np.array_equal(block, np.asarray(array[:, start:stop]).T)):
                raise ValueError("invalid canonical affinity numerical block")
    return array


def scratch_target_valid(path: Path) -> bool:
    target = path.resolve()
    return target.parent.is_dir() and not target.is_relative_to(REPO_ROOT.resolve())


def preflight_full(affinity_dir: Path, notebook_path: Path,
                   output_dir: Path | None, technical_output: bool) -> dict[str, object]:
    validate_inputs(affinity_dir, notebook_path, full_scan=True)
    total, available = reference.physical_memory()
    disk_target = output_dir.parent if output_dir is not None else affinity_dir
    while not disk_target.exists():
        disk_target = disk_target.parent
    free_disk = shutil.disk_usage(disk_target).free
    path_valid = output_dir is not None and (
        scratch_target_valid(output_dir) if technical_output
        else output_dir.resolve() == OFFICIAL_OUTPUT.resolve()
    )
    collision = False
    if output_dir is not None:
        try:
            reference.refuse_output_target(output_dir)
        except ValueError:
            collision = True
    return {
        "mode": "preflight-full", "technical_output": technical_output,
        "environment": reference.verify_environment(),
        "input_affinity_sha256": AFFINITY_SHA, "input_rows_sha256": ROWS_SHA,
        "input_affinity_spec_sha256": SPEC_SHA, "source_H_sha256": SOURCE_H_SHA,
        "iic_final_sha256": IIC_FINAL_SHA, "notebook06_sha256": NOTEBOOK_SHA,
        "affinity_bytes": N * N * 8,
        "total_physical_ram_bytes": total, "available_ram_bytes": available,
        "free_disk_bytes": free_disk, "estimated_peak_ram_bytes": ESTIMATED_PEAK_RAM,
        "minimum_available_ram_bytes": MIN_AVAILABLE_RAM,
        "minimum_free_disk_bytes": MIN_FREE_DISK,
        "output_collision": collision, "output_path_valid": path_valid,
        "full_ready": (available >= MIN_AVAILABLE_RAM and free_disk >= MIN_FREE_DISK
                       and not collision and path_valid),
        "eigsh_calls": 0, "kmeans_fits": 0, "output_created": False,
    }


def committed_producer() -> str:
    status = subprocess.check_output(
        ["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT, text=True,
    )
    if status.strip():
        raise ValueError("official spectral producer requires clean Git")
    commit = subprocess.check_output(
        ["git", "--no-optional-locks", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
    ).strip()
    relative = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "--no-optional-locks", "show", f"HEAD:{relative}"], cwd=REPO_ROOT,
    )
    if hashlib.sha256(committed).hexdigest() != reference.sha256_file(Path(__file__)):
        raise ValueError("official spectral producer is not byte-identical to HEAD")
    return commit


def validate_result(result: dict[str, object]) -> None:
    expected = {
        "eigenvalues": (K,), "eigenvectors": (N, K),
        "embedding": (N, K), "labels": (N,),
    }
    for name, shape in expected.items():
        array = np.asarray(result[name])
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"spectral {name} shape/finite mismatch")
    labels = np.asarray(result["labels"])
    if np.any(labels < 0) or np.any(labels >= K):
        raise ValueError("spectral label range mismatch")


def publish_full(output_dir: Path, affinity_dir: Path, result: dict[str, object],
                 versions: dict[str, str], technical: bool) -> None:
    validate_result(result)
    reference.refuse_output_target(output_dir)
    if technical:
        if not scratch_target_valid(output_dir):
            raise ValueError("technical full output must be outside repository")
        commit = "UNCOMMITTED_WIP"
    else:
        if output_dir.resolve() != OFFICIAL_OUTPUT.resolve():
            raise ValueError("official spectral output identity mismatch")
        commit = committed_producer()
    partial = Path(str(output_dir) + ".partial")
    partial.mkdir(parents=False, exist_ok=False)
    arrays = {
        "eigenvalues_f64.bin": (result["eigenvalues"], "<f8"),
        "eigenvectors_f64.bin": (result["eigenvectors"], "<f8"),
        "spectral_embedding_f64.bin": (result["embedding"], "<f8"),
        "cluster_labels_i32.bin": (result["labels"], "<i4"),
    }
    for name, (array, dtype) in arrays.items():
        np.ascontiguousarray(array, dtype=dtype).tofile(partial / name)
    shutil.copyfile(affinity_dir / "rows.csv", partial / "rows.csv")
    environment = {
        **versions, "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(), "threadpools": threadpoolctl.threadpool_info(),
    }
    (partial / "environment.txt").write_text(
        json.dumps(environment, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    spec = {
        "format": "VAE_C_IIC_SPECTRAL_ARTIFACT_V1", "rows": N, "components": K,
        "clusters": K, "dtype_eigensystem": "float64_le", "dtype_labels": "int32_le",
        "layout": "row_major", "affinity_sha256": AFFINITY_SHA,
        "affinity_rows_sha256": ROWS_SHA, "notebook06_sha256": NOTEBOOK_SHA,
        "source_H_sha256": SOURCE_H_SHA, "iic_final_sha256": IIC_FINAL_SHA,
        "iic_run_id": RUN_ID, "eigensolver": "scipy.sparse.linalg.eigsh",
        "eigensolver_k": K, "eigensolver_which": "LA", "eigensolver_tol": 1e-8,
        "eigensolver_maxiter": 100000, "eigensolver_ncv": 145,
        "v0": "default_rng(123).standard_normal(n).astype(float64); divide_by_l2_norm",
        "eigenvalue_order": "np.argsort(values)[::-1]",
        "sign_fix": "first_np_argmax_abs_positive", "row_normalization": "L2",
        "kmeans_n_clusters": K, "kmeans_init": "k-means++", "kmeans_seed": 123,
        "kmeans_n_init": 50, "kmeans_max_iter": 1000, "kmeans_tol": 1e-6,
        "kmeans_algorithm": "lloyd", "kmeans_inertia": result["kmeans_inertia"],
        "kmeans_iterations": result["kmeans_iterations"],
        "residual_max": result["residual_max"],
        "orthogonality_max": result["orthogonality_max"], "label_free": True,
        "producer_commit": commit, "producer_script_sha256": reference.sha256_file(Path(__file__)),
        "environment": versions, "temporary_M_created": False,
        "technical_only": technical,
    }
    (partial / "spectral_spec.txt").write_text(
        json.dumps(spec, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    names = sorted(path.name for path in partial.iterdir() if path.is_file())
    manifest = "".join(
        f"{reference.sha256_file(partial / name)}  {name}\n" for name in names
    )
    (partial / "sha256.txt").write_text(manifest, encoding="ascii")
    expected = set(arrays) | {"rows.csv", "environment.txt", "spectral_spec.txt", "sha256.txt"}
    if {path.name for path in partial.iterdir()} != expected:
        raise ValueError("spectral V1 file inventory mismatch")
    for line in manifest.splitlines():
        sha, name = line.split("  ", 1)
        if reference.sha256_file(partial / name) != sha:
            raise ValueError("spectral publish checksum mismatch")
    partial.rename(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=(
        "technical-golden", "technical-subset", "preflight-full", "technical-full", "run-full",
    ))
    parser.add_argument("--affinity-dir", type=Path, default=AFFINITY_DIR)
    parser.add_argument("--notebook-path", type=Path, default=reference.REFERENCE_NOTEBOOK)
    parser.add_argument("--subset-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scratch-output", action="store_true")
    parser.add_argument("--confirm-full-run", action="store_true")
    args = parser.parse_args()
    verify_reference()
    versions = reference.verify_environment()
    reference.verify_file(args.notebook_path, NOTEBOOK_SIZE, NOTEBOOK_SHA)
    if args.mode == "technical-golden":
        rng = np.random.default_rng(20260917)
        h = rng.standard_normal((64, 5))
        affinity = np.exp(-np.linalg.norm(h[:, None, :] - h[None, :, :], axis=2))
        report = reference.technical_report(affinity, args.mode)
    elif args.mode == "technical-subset":
        if args.subset_size < K + 2 or args.subset_size > 512:
            raise ValueError("technical subset size must be 38..512")
        canonical = validate_inputs(args.affinity_dir, args.notebook_path, full_scan=False)
        indices = np.linspace(0, N - 1, args.subset_size, dtype=np.int64)
        affinity = np.asarray(canonical[np.ix_(indices, indices)])
        report = reference.technical_report(
            affinity, args.mode, reference.array_sha256(indices, "<i8"),
        )
    elif args.mode == "preflight-full":
        report = preflight_full(
            args.affinity_dir, args.notebook_path, args.output_dir, args.scratch_output,
        )
    else:
        technical = args.mode == "technical-full"
        if not args.confirm_full_run or args.output_dir is None:
            raise ValueError("full run requires --confirm-full-run and --output-dir")
        if technical != args.scratch_output:
            raise ValueError("technical-full requires --scratch-output; run-full forbids it")
        if not technical:
            committed_producer()
        preflight = preflight_full(
            args.affinity_dir, args.notebook_path, args.output_dir, technical,
        )
        if not preflight["full_ready"]:
            raise ValueError("full resource/input/output preflight not ready")
        canonical = np.memmap(
            args.affinity_dir / "affinity_f64.bin", dtype="<f8", mode="r", shape=(N, N),
        )
        result = reference.spectral_chain(canonical)
        publish_full(args.output_dir, args.affinity_dir, result, versions, technical)
        report = {"mode": args.mode, "technical_only": technical, "label_free": True,
                  "output_created": True, "output_dir": str(args.output_dir),
                  "kmeans_iterations": result["kmeans_iterations"],
                  "kmeans_inertia": result["kmeans_inertia"],
                  "residual_max": result["residual_max"],
                  "orthogonality_max": result["orthogonality_max"]}
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"SPECTRAL_BRIDGE_ERROR={error}", file=sys.stderr)
        sys.exit(2)
