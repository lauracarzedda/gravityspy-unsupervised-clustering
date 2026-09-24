"""Label-free, non-full technical gates for the mini224 spectral bridge.

Fixtures are confined to the external build tree supplied by CTest.  The
canonical affinity is read only, and no official spectral output is created.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def rejects(action, message: str) -> None:
    try:
        action()
    except ValueError:
        return
    raise AssertionError(message)


def invoke(producer: Path, *arguments: str) -> tuple[bytes, dict[str, object]]:
    environment = os.environ.copy()
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    process = subprocess.run(
        [sys.executable, "-B", str(producer), *arguments], env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=300,
    )
    require(process.returncode == 0, process.stderr.decode(errors="replace"))
    return process.stdout, json.loads(process.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    producer = args.producer.resolve(strict=True)
    repo_root = args.repo_root.resolve(strict=True)
    fixture_root = args.fixture_root.resolve()
    require(not fixture_root.is_relative_to(repo_root), "fixture root inside repository")
    fixture_root.mkdir(parents=True, exist_ok=True)
    fixture = Path(tempfile.mkdtemp(prefix="sp224_", dir=fixture_root))

    ast.parse(producer.read_text(encoding="utf-8"), filename=str(producer))
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(producer.parent))
    import run_vae_c_iic_spectral_mini224 as bridge

    bridge.verify_reference()
    versions = bridge.reference.verify_environment()
    require(versions["numpy"] == importlib.metadata.version("numpy"), "NumPy identity")
    require((bridge.N, bridge.K) == (1000, 36), "mini224 dimensions")
    require(bridge.AFFINITY_DIR.is_dir(), "canonical affinity missing")
    official_output_existed = bridge.OFFICIAL_OUTPUT.exists()
    bridge.validate_inputs(bridge.AFFINITY_DIR, bridge.reference.REFERENCE_NOTEBOOK, True)

    golden_a, golden_report = invoke(producer, "--mode", "technical-golden")
    golden_b, _ = invoke(producer, "--mode", "technical-golden")
    require(golden_a == golden_b, "fresh-process golden differs")
    require(golden_report["eigenvalue_max_abs_vs_dense"] < 1e-8
            and golden_report["eigenspace_projector_max_abs_vs_dense"] < 1e-7,
            "golden dense reference")
    subset_a, subset_report = invoke(producer, "--mode", "technical-subset",
                                     "--subset-size", "256")
    subset_b, _ = invoke(producer, "--mode", "technical-subset",
                                     "--subset-size", "256")
    require(subset_a == subset_b, "fresh-process subset differs")
    require(subset_report["eigenvalue_max_abs_vs_dense"] < 1e-8
            and subset_report["eigenspace_projector_max_abs_vs_dense"] < 1e-7,
            "subset dense reference")

    target = fixture / "preflight_absent"
    _, preflight = invoke(producer, "--mode", "preflight-full", "--scratch-output",
                          "--output-dir", str(target))
    require(preflight["eigsh_calls"] == preflight["kmeans_fits"] == 0,
            "preflight ran spectral computation")
    require(not preflight["output_created"] and not target.exists(),
            "preflight created output")
    require(preflight["output_path_valid"] and not preflight["output_collision"],
            "scratch preflight path invalid")

    rows = bridge.AFFINITY_DIR / "rows.csv"
    malformed = fixture / "rows.csv"
    malformed.write_text(
        "manifest_index,event_index,event_id,source_split,split_pos\n"
        "0,1,TECHNICAL,train,0\n", encoding="utf-8", newline="\n",
    )
    rejects(lambda: bridge.validate_rows(malformed), "short row map accepted")
    rejects(lambda: bridge.reference.verify_file(rows, rows.stat().st_size, "0" * 64),
            "row SHA mismatch accepted")
    rejects(lambda: bridge.reference.verify_file(bridge.AFFINITY_DIR / "affinity_spec.txt",
                                                 1379, "0" * 64),
            "spec SHA mismatch accepted")
    rejects(lambda: bridge.reference.verify_file(bridge.AFFINITY_DIR / "affinity_f64.bin",
                                                 8_000_001, bridge.AFFINITY_SHA),
            "affinity size mismatch accepted")
    expected_python = bridge.reference.REFERENCE_PYTHON
    try:
        bridge.reference.REFERENCE_PYTHON = fixture / "incompatible_python.exe"
        rejects(bridge.reference.verify_environment, "incompatible environment accepted")
    finally:
        bridge.reference.REFERENCE_PYTHON = expected_python
    invalid = np.eye(64, dtype=np.float64)
    invalid[0, 1] = invalid[1, 0] = np.nan
    rejects(lambda: bridge.reference.technical_report(invalid, "technical-golden"),
            "NaN affinity accepted")
    invalid[0, 1] = invalid[1, 0] = np.inf
    rejects(lambda: bridge.reference.technical_report(invalid, "technical-golden"),
            "Inf affinity accepted")
    require(not bridge.scratch_target_valid(bridge.OFFICIAL_OUTPUT),
            "repository output accepted as scratch")
    target.mkdir()
    collision = bridge.preflight_full(bridge.AFFINITY_DIR,
                                      bridge.reference.REFERENCE_NOTEBOOK, target, True)
    require(collision["output_collision"] and not collision["full_ready"],
            "existing output collision accepted")
    rejects(lambda: bridge.validate_result({
        "eigenvalues": np.zeros(35), "eigenvectors": np.zeros((1000, 36)),
        "embedding": np.zeros((1000, 36)), "labels": np.zeros(1000),
    }), "component mismatch accepted")
    rejects(lambda: bridge.validate_result({
        "eigenvalues": np.zeros(36), "eigenvectors": np.zeros((999, 36)),
        "embedding": np.zeros((1000, 36)), "labels": np.zeros(1000),
    }), "N mismatch accepted")
    require(bridge.OFFICIAL_OUTPUT.exists() == official_output_existed,
            "test changed official output presence")
    print("MINI224_SPECTRAL_BRIDGE_TEST_PASS=true")
    print("GOLDEN_STDOUT_SHA256=" + hashlib.sha256(golden_a).hexdigest())
    print("SUBSET_STDOUT_SHA256=" + hashlib.sha256(subset_a).hexdigest())
    print("PREFLIGHT_FULL_READY=" + str(preflight["full_ready"]).lower())


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, ValueError, RuntimeError) as error:
        print(f"MINI224_SPECTRAL_BRIDGE_TEST_PASS=false: {error}", file=sys.stderr)
        raise SystemExit(1) from error
