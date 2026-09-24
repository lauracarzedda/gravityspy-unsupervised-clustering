"""Non-full, label-free validation of the mini75 IIC spectral bridge.

All generated fixtures stay in the caller-supplied external build tree.  The
canonical affinity and row map are read only; this test never runs the 8583-row
eigensolver or creates a scientific spectral artifact.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np


EXPECTED_VERSIONS = {
    "numpy": "2.5.3",
    "scipy": "1.16.3",
    "scikit-learn": "1.6.1",
    "joblib": "1.6.0",
    "threadpoolctl": "3.7.0",
}
EXPECTED_AFFINITY_BYTES = 589343112
EXPECTED_AFFINITY_SHA256 = (
    "c3bd99194614bbb69557327072168f0293819424dad272a8debb88d48e794058"
)
EXPECTED_ROWS_SHA256 = (
    "6e09d72e3b54f213a2096edffca627fd35f1f0e35778539f05a810a788d7c642"
)
EXPECTED_SUBSET_INDEX_SHA256 = (
    "47b7a1943a719a040907a77912de7702686cf17c29b00972ad44604bbde879de"
)
EXPECTED_SUBSET_INERTIA = 23.476957030837784
MIN_AVAILABLE_RAM_BYTES = 4_000_000_000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def raises_value_error(operation, description: str) -> None:
    try:
        operation()
    except ValueError:
        return
    raise AssertionError(f"{description}: ValueError was not raised")


def load_producer(path: Path):
    # Do not use py_compile: it would write __pycache__ beside the source.
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("vae_c_iic_spectral_bridge_tested", path)
    require(spec is not None and spec.loader is not None, "cannot import producer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_environment(module) -> None:
    require(platform.python_version() == "3.12.14", "reference Python version mismatch")
    for package, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(package)
        require(actual == expected, f"{package} version {actual} != {expected}")
    require(callable(module.verify_environment), "verify_environment unavailable")
    module.verify_environment()


def canonical_affinity_directory(repo_root: Path) -> Path:
    root = repo_root / "sakai_multires_c" / "results" / "vae_c" / "iic"
    matches = sorted(root.glob("mini75_iic_v1_*_affinity_primary_v1"))
    require(len(matches) == 1, f"expected one canonical affinity directory, found {len(matches)}")
    return matches[0]


def check_input_helpers(module, fixture: Path, affinity_dir: Path) -> None:
    require(callable(module.verify_file), "verify_file unavailable")
    require(callable(module.validate_rows), "validate_rows unavailable")
    require(callable(module.refuse_output_target), "refuse_output_target unavailable")

    affinity = affinity_dir / "affinity_f64.bin"
    rows = affinity_dir / "rows.csv"
    require(affinity.stat().st_size == EXPECTED_AFFINITY_BYTES, "canonical affinity byte count")
    module.verify_file(affinity, EXPECTED_AFFINITY_BYTES, EXPECTED_AFFINITY_SHA256)
    module.verify_file(rows, rows.stat().st_size, EXPECTED_ROWS_SHA256)
    module.validate_rows(rows, 8583)

    tiny = fixture / "tiny_f64.bin"
    tiny.write_bytes(bytes.fromhex("000000000000f03f0000000000000000"))
    tiny_sha = hashlib.sha256(tiny.read_bytes()).hexdigest()
    module.verify_file(tiny, 16, tiny_sha)
    raises_value_error(lambda: module.verify_file(tiny, 17, tiny_sha), "wrong byte size")
    raises_value_error(lambda: module.verify_file(tiny, 16, "0" * 64), "wrong SHA-256")
    raises_value_error(lambda: module.verify_file(fixture / "absent.bin", 16, tiny_sha), "missing file")

    malformed_rows = fixture / "malformed_rows.csv"
    malformed_rows.write_text(
        "manifest_index,event_index,event_id,source_split,split_pos\n"
        "0,1,TECHNICAL,train,0\n",
        encoding="utf-8",
        newline="\n",
    )
    raises_value_error(lambda: module.validate_rows(malformed_rows, 8583), "row-map truncation")

    nonfinite = np.eye(64, dtype=np.float64)
    nonfinite[0, 1] = nonfinite[1, 0] = np.nan
    raises_value_error(lambda: module.technical_report(nonfinite, "technical-golden"), "NaN affinity")
    negative = np.eye(64, dtype=np.float64)
    negative[0, 1] = negative[1, 0] = -0.1
    raises_value_error(lambda: module.technical_report(negative, "technical-golden"), "negative affinity")
    above_one = np.eye(64, dtype=np.float64)
    above_one[0, 1] = above_one[1, 0] = 1.1
    raises_value_error(lambda: module.technical_report(above_one, "technical-golden"), "affinity above one")

    unused_target = fixture / "unused_output"
    module.refuse_output_target(unused_target)
    existing_target = fixture / "existing_output"
    existing_target.mkdir()
    raises_value_error(lambda: module.refuse_output_target(existing_target), "overwrite refusal")
    partial_target = fixture / "partial_output"
    Path(str(partial_target) + ".partial").mkdir()
    raises_value_error(lambda: module.refuse_output_target(partial_target), "partial overwrite refusal")
    tmp_target = fixture / "tmp_output"
    Path(str(tmp_target) + ".tmp").mkdir()
    raises_value_error(lambda: module.refuse_output_target(tmp_target), "tmp overwrite refusal")


def run_producer(producer: Path, mode: str, affinity_dir: Path | None = None,
                 output_dir: Path | None = None) -> tuple[int, bytes, bytes, dict]:
    command = [sys.executable, "-B", str(producer), "--mode", mode]
    if affinity_dir is not None:
        command += ["--affinity-dir", str(affinity_dir)]
    if output_dir is not None:
        command += ["--output-dir", str(output_dir)]
    if mode == "technical-subset":
        command += ["--subset-size", "256"]
    environment = os.environ.copy()
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    process = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=environment, check=False, timeout=600,
    )
    try:
        report = json.loads(process.stdout)
    except (ValueError, UnicodeDecodeError) as exc:
        raise AssertionError(
            f"{mode}: stdout is not one JSON object; exit={process.returncode}; "
            f"stderr={process.stderr.decode('utf-8', errors='replace')}"
        ) from exc
    require(isinstance(report, dict), f"{mode}: JSON root must be an object")
    return process.returncode, process.stdout, process.stderr, report


def expect_cli_rejection(
    producer: Path, *, affinity_dir: Path | None, notebook_path: Path | None,
    output_dir: Path, expected_error: str,
) -> None:
    command = [sys.executable, "-B", str(producer), "--mode", "technical-subset",
               "--subset-size", "256", "--output-dir", str(output_dir)]
    if affinity_dir is not None:
        command += ["--affinity-dir", str(affinity_dir)]
    if notebook_path is not None:
        command += ["--notebook-path", str(notebook_path)]
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             env=environment, check=False, timeout=120)
    error = process.stderr.decode("utf-8", errors="replace")
    require(process.returncode != 0, f"{expected_error}: malformed CLI input was accepted")
    require(process.stdout == b"", f"{expected_error}: unexpected producer report")
    require(expected_error in error, f"expected {expected_error!r}, got {error!r}")
    require(not output_dir.exists() and not Path(str(output_dir) + ".partial").exists(),
            f"{expected_error}: created output")


def check_cli_input_rejection(producer: Path, fixture: Path, affinity_dir: Path) -> None:
    short_dir = fixture / "wrong_size_affinity"
    short_dir.mkdir()
    (short_dir / "affinity_f64.bin").write_bytes(b"invalid")
    expect_cli_rejection(producer, affinity_dir=short_dir, notebook_path=None,
                         output_dir=fixture / "wrong_size_output",
                         expected_error="missing or wrong-size file")

    wrong_sha_dir = fixture / "wrong_sha_affinity"
    wrong_sha_dir.mkdir()
    wrong_sha_payload = wrong_sha_dir / "affinity_f64.bin"
    # A sparse/zero-filled fixture has the expected size but the wrong digest.
    # It is external to the repository and removed after the negative gate.
    try:
        with wrong_sha_payload.open("wb") as stream:
            stream.truncate(EXPECTED_AFFINITY_BYTES)
        expect_cli_rejection(producer, affinity_dir=wrong_sha_dir, notebook_path=None,
                             output_dir=fixture / "wrong_sha_output",
                             expected_error="SHA-256 mismatch")
    finally:
        wrong_sha_payload.unlink(missing_ok=True)

    wrong_notebook = fixture / "wrong_notebook.ipynb"
    with wrong_notebook.open("wb") as stream:
        stream.truncate(1_220_480)
    expect_cli_rejection(producer, affinity_dir=affinity_dir, notebook_path=wrong_notebook,
                         output_dir=fixture / "wrong_notebook_output",
                         expected_error="SHA-256 mismatch")


def check_success_report(mode: str, code: int, stderr: bytes, report: dict) -> None:
    require(code == 0, f"{mode}: exit={code}, stderr={stderr.decode('utf-8', errors='replace')}")
    require(report.get("mode") == mode, f"{mode}: mode identity mismatch")
    require(report.get("technical_only") is True, f"{mode}: technical-only gate missing")
    require(report.get("label_free") is True, f"{mode}: label-free gate missing")
    require(report.get("output_created") is False, f"{mode}: output created")
    require(report.get("k") == 36, f"{mode}: K mismatch")
    require(report.get("W_diagonal_zero") is True, f"{mode}: W diagonal policy")
    require(isinstance(report.get("cluster_min"), int) and report["cluster_min"] > 0,
            f"{mode}: empty cluster")
    require(isinstance(report.get("residual_max"), float) and report["residual_max"] <= 1e-6,
            f"{mode}: eigen residual")
    require(isinstance(report.get("orthogonality_max"), float)
            and report["orthogonality_max"] <= 1e-8, f"{mode}: orthogonality")
    require(isinstance(report.get("kmeans_inertia"), float)
            and np.isfinite(report["kmeans_inertia"]), f"{mode}: KMeans inertia")


def check_non_full_pipeline(producer: Path, affinity_dir: Path) -> tuple[str, str]:
    digests = []
    for mode, input_dir in (
        ("technical-golden", None),
        ("technical-subset", affinity_dir),
    ):
        first = run_producer(producer, mode, input_dir)
        second = run_producer(producer, mode, input_dir)
        check_success_report(mode, first[0], first[2], first[3])
        check_success_report(mode, second[0], second[2], second[3])
        require(first[1] == second[1], f"{mode}: fresh-process stdout differs")
        expected_rows = 64 if mode == "technical-golden" else 256
        require(first[3].get("rows") == expected_rows, f"{mode}: row count")
        if mode == "technical-golden":
            require(first[3].get("eigenvector_max_abs_vs_dense", 1.0) <= 1e-8,
                    "synthetic golden eigenvectors")
            require(first[3].get("embedding_max_abs_vs_dense", 1.0) <= 1e-8,
                    "synthetic golden embedding")
        else:
            require(first[3].get("subset_index_sha256") == EXPECTED_SUBSET_INDEX_SHA256,
                    "technical subset index identity mismatch")
            require(abs(first[3]["kmeans_inertia"] - EXPECTED_SUBSET_INERTIA) <= 1e-9,
                    "technical subset inertia differs from the validated reference")
        digests.append(hashlib.sha256(first[1]).hexdigest())
    return digests[0], digests[1]


def check_full_preflight(producer: Path, affinity_dir: Path, fixture: Path) -> dict:
    before = tuple(sorted(item.name for item in fixture.iterdir()))
    # The bridge requires the target to be a sibling of its affinity input.
    # Use a unique technical name, never the frozen scientific output name.
    planned = affinity_dir.parent / ("__spectral_preflight_test_" + fixture.name)
    require(all(not Path(str(planned) + suffix).exists()
                for suffix in ("", ".partial", ".tmp")),
            "technical preflight output target already exists")
    code, _stdout, stderr, report = run_producer(producer, "preflight-full", affinity_dir, planned)
    require(code == 0, f"preflight-full exit={code}: {stderr.decode('utf-8', errors='replace')}")
    require(report.get("mode") == "preflight-full", "preflight mode identity")
    require(report.get("eigsh_calls") == 0, "full eigsh ran during preflight")
    require(report.get("kmeans_fits") == 0, "KMeans ran during preflight")
    require(report.get("output_created") is False, "preflight created output")
    require(report.get("output_path_valid") is True, "planned full output path invalid")
    available = report.get("available_ram_bytes")
    require(isinstance(available, int) and available >= 0, "available RAM missing")
    free_disk = report.get("free_disk_bytes")
    require(isinstance(free_disk, int) and free_disk >= 0, "free disk missing")
    expected_ready = (available >= MIN_AVAILABLE_RAM_BYTES and free_disk >= 32_000_000
                      and report.get("output_collision") is False)
    require(report.get("full_ready") is expected_ready, "resource/output gate is incorrect")
    require(tuple(sorted(item.name for item in fixture.iterdir())) == before, "preflight changed fixture")
    require(all(not Path(str(planned) + suffix).exists()
                for suffix in ("", ".partial", ".tmp")),
            "preflight created technical output")
    return report


def check_full_requires_confirmation(producer: Path, fixture: Path) -> None:
    forbidden = fixture / "forbidden_full_output"
    process = subprocess.run(
        [sys.executable, "-B", str(producer), "--mode", "run-full",
         "--output-dir", str(forbidden)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30,
    )
    require(process.returncode != 0, "full run accepted without confirmation")
    require(b"full run requires --confirm-full-run" in process.stderr,
            "full run rejected for an unexpected reason")
    require(not forbidden.exists() and not Path(str(forbidden) + ".partial").exists(),
            "unconfirmed full run created output")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    producer = args.producer.resolve(strict=True)
    repo_root = args.repo_root.resolve(strict=True)
    fixture_root = args.fixture_root.resolve()
    results_root = repo_root / "sakai_multires_c" / "results"
    require(not fixture_root.is_relative_to(results_root), "fixture root must be outside results")
    fixture_root.mkdir(parents=True, exist_ok=True)
    fixture = Path(tempfile.mkdtemp(prefix="sp8b_", dir=fixture_root))
    affinity_dir = canonical_affinity_directory(repo_root)

    module = load_producer(producer)
    check_environment(module)
    check_input_helpers(module, fixture, affinity_dir)
    check_cli_input_rejection(producer, fixture, affinity_dir)
    golden_sha, subset_sha = check_non_full_pipeline(producer, affinity_dir)
    preflight = check_full_preflight(producer, affinity_dir, fixture)
    check_full_requires_confirmation(producer, fixture)

    print("SPECTRAL_BRIDGE_TEST_PASS=true")
    print(f"SPECTRAL_GOLDEN_STDOUT_SHA256={golden_sha}")
    print(f"SPECTRAL_SUBSET_STDOUT_SHA256={subset_sha}")
    print(f"SPECTRAL_PREFLIGHT_AVAILABLE_RAM_BYTES={preflight['available_ram_bytes']}")
    print(f"SPECTRAL_PREFLIGHT_FULL_READY={str(preflight['full_ready']).lower()}")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, ValueError, RuntimeError) as exc:
        print(f"SPECTRAL_BRIDGE_TEST_PASS=false: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
