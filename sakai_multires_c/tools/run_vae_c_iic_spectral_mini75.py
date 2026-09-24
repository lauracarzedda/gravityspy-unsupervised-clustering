"""Label-free mini75 IIC spectral bridge to the Notebook06 Python reference.

Technical modes never publish artifacts.  The full mode requires an explicit
confirmation, clean committed producer, valid canonical inputs, and a resource
gate; it is not invoked by the tests.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.cluster import KMeans


N = 8583
K = 36
NCV = 145
PEAK_RAM_BUDGET = 2_500_000_000
MIN_AVAILABLE_RAM = 4_000_000_000
MIN_FREE_DISK = 32_000_000
TEMP_M_BYTES = 0  # Notebook06 S5 uses a LinearOperator, not a stored M.
ESTIMATED_STAGING_DISK_BYTES = 5_300_000  # Outputs in .partial; no NxN temporary.
RUN_ID = "mini75_iic_v1_737da555866b822e14c545c4f2687869f41a52cd1340b9e941f3a1ce77763eb9"
REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PYTHON = Path(sys.executable)
REFERENCE_NOTEBOOK = Path(
    REPO_ROOT
    / "notebooks"
    / "full224"
    / "06_sakai224_full_iic_and_spectral.ipynb"
)
NOTEBOOK_SIZE = 1_220_480
NOTEBOOK_SHA = "d42d3f807a46aefadf933ba584e06891e042a4efd38b1248e07f0c660b563495"
NOTEBOOK_HISTORICAL_SHA = "81894341420d94c993c916663e4ce83578f53734d7d1c35f9c0425e3b4198c8c"
AFFINITY_SHA = "c3bd99194614bbb69557327072168f0293819424dad272a8debb88d48e794058"
ROWS_SHA = "6e09d72e3b54f213a2096edffca627fd35f1f0e35778539f05a810a788d7c642"
SPEC_SHA = "3f172cda943dcc05d882e09945b65236cb04a4a2db633493f84939b09b3528d8"
CHECKSUM_SHA = "8663181e1659338aa21a418760cf8d6bd8270dcc162252f5e581eafb23dfb1e6"
SOURCE_H_SHA = "1807ea44b7e3c884b04f7a35195731806916374caf8297e0294e7aaa55abeb44"
IIC_FINAL_SHA = "6df9bb278210861b449adec454c4f649fee8be1398082418096c2679aee907d5"
DEFAULT_AFFINITY_DIR = (
    REPO_ROOT / "sakai_multires_c" / "results" / "vae_c" / "iic"
    / (RUN_ID + "_affinity_primary_v1")
)
ROW_HEADER = ["manifest_index", "event_index", "event_id", "source_split", "split_pos"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def verify_file(path: Path, expected_size: int, expected_sha256: str) -> None:
    path = Path(path)
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError(f"missing or wrong-size file: {path}")
    if sha256_file(path) != expected_sha256.lower():
        raise ValueError(f"SHA-256 mismatch: {path}")


def verify_environment() -> dict[str, str]:
    actual = os.path.normcase(os.path.realpath(sys.executable))
    expected = os.path.normcase(os.path.realpath(REFERENCE_PYTHON))
    if actual != expected:
        raise ValueError(f"wrong Python interpreter: {sys.executable}")
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
    }
    expected_versions = {
        "python": "3.12.14", "numpy": "2.5.3", "scipy": "1.16.3",
        "sklearn": "1.6.1", "joblib": "1.6.0", "threadpoolctl": "3.7.0",
    }
    if versions != expected_versions:
        raise ValueError(f"reference environment mismatch: {versions}")
    return versions


def validate_rows(path: Path, expected_n: int) -> dict[str, int]:
    seen_ids: set[str] = set()
    split_positions = {"train": set(), "validation": set()}
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ROW_HEADER:
            raise ValueError("invalid label-free row-map header")
        for position, row in enumerate(reader):
            if set(row) != set(ROW_HEADER):
                raise ValueError("malformed row-map record")
            try:
                manifest_index = int(row["manifest_index"])
                event_index = int(row["event_index"])
                split_pos = int(row["split_pos"])
            except (TypeError, ValueError) as error:
                raise ValueError("non-integer row identity") from error
            split = row["source_split"]
            event_id = row["event_id"]
            if (position >= expected_n or manifest_index != position
                    or event_index != position + 1 or not event_id
                    or event_id in seen_ids or split not in split_positions
                    or split_pos < 0 or split_pos in split_positions[split]):
                raise ValueError("row-map identity/order mismatch")
            seen_ids.add(event_id)
            split_positions[split].add(split_pos)
    if len(seen_ids) != expected_n:
        raise ValueError("row-map count mismatch")
    if expected_n == N:
        for split, count in (("train", 6866), ("validation", 1717)):
            if split_positions[split] != set(range(count)):
                raise ValueError(f"{split} split positions mismatch")
    return {split: len(positions) for split, positions in split_positions.items()}


def validate_inputs(directory: Path, notebook_path: Path, full_scan: bool) -> np.memmap:
    directory = Path(directory)
    verify_file(directory / "affinity_f64.bin", N * N * 8, AFFINITY_SHA)
    verify_file(directory / "rows.csv", 304612, ROWS_SHA)
    verify_file(directory / "affinity_spec.txt", 1379, SPEC_SHA)
    verify_file(directory / "sha256.txt", 242, CHECKSUM_SHA)
    verify_file(Path(notebook_path), NOTEBOOK_SIZE, NOTEBOOK_SHA)
    verify_file(directory.parent / (RUN_ID + "_h_main_v1") / "h_f32.bin", 6179760, SOURCE_H_SHA)
    verify_file(directory.parent / RUN_ID / "final.ckpt", 10935035, IIC_FINAL_SHA)
    names = {p.name for p in directory.iterdir() if p.is_file()}
    if names != {"affinity_f64.bin", "rows.csv", "affinity_spec.txt", "sha256.txt"}:
        raise ValueError("unexpected or missing canonical affinity file")
    spec = {}
    for line in (directory / "affinity_spec.txt").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in spec:
            raise ValueError("malformed affinity spec")
        spec[key] = value
    required = {
        "format": "VAE_C_IIC_AFFINITY_ARTIFACT_V1", "rows": str(N),
        "cols": str(N), "dtype": "float64_ieee754", "endian": "little",
        "layout": "row_major", "payload_sha256": AFFINITY_SHA,
        "row_map_sha256": ROWS_SHA, "diagonal": "exact_1.0",
        "symmetry": "upper_once_lower_exact_copy", "label_free": "true",
        "notebook06_reference_sha256": NOTEBOOK_HISTORICAL_SHA,
    }
    if any(spec.get(key) != value for key, value in required.items()):
        raise ValueError("affinity spec identity mismatch")
    validate_rows(directory / "rows.csv", N)
    array = np.memmap(directory / "affinity_f64.bin", dtype="<f8", mode="r", shape=(N, N))
    if full_scan:
        for start in range(0, N, 128):
            stop = min(N, start + 128)
            block = np.asarray(array[start:stop, :])
            if (not np.isfinite(block).all() or np.any(block < 0.0)
                    or np.any(block > 1.0)
                    or not np.array_equal(np.diag(block[:, start:stop]), np.ones(stop - start))
                    or not np.array_equal(block, np.asarray(array[:, start:stop]).T)):
                raise ValueError(f"invalid affinity numerical block: {start}:{stop}")
    return array


def fixed_v0(n: int) -> np.ndarray:
    values = np.random.default_rng(123).standard_normal(n).astype(np.float64)
    return values / np.linalg.norm(values)


def normalized_operator(affinity: np.ndarray) -> tuple[np.ndarray, LinearOperator]:
    n = affinity.shape[0]
    if affinity.ndim != 2 or affinity.shape != (n, n) or n <= K:
        raise ValueError("invalid affinity shape")
    if not np.array_equal(np.diag(affinity), np.ones(n)):
        raise ValueError("affinity diagonal must be exactly one")
    degree = affinity.sum(axis=1, dtype=np.float64) - 1.0
    if not np.isfinite(degree).all() or np.any(degree <= 0.0):
        raise ValueError("invalid affinity degree")
    inv = 1.0 / np.sqrt(degree)

    def matvec(x: np.ndarray) -> np.ndarray:
        z = inv * x
        return inv * (affinity @ z - z)

    def matmat(x: np.ndarray) -> np.ndarray:
        z = inv[:, None] * x
        return inv[:, None] * (affinity @ z - z)

    return degree, LinearOperator((n, n), matvec=matvec, matmat=matmat, dtype=np.float64)


def spectral_chain(affinity: np.ndarray) -> dict[str, object]:
    n = affinity.shape[0]
    degree, operator = normalized_operator(affinity)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        eigenvalues, eigenvectors = eigsh(
            operator, k=K, which="LA", tol=1e-8, maxiter=100000,
            ncv=min(NCV, n), v0=fixed_v0(n),
        )
    if emitted:
        raise ValueError(f"eigsh warning: {emitted[0].message}")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    for column in range(K):
        pivot = int(np.argmax(np.abs(eigenvectors[:, column])))
        if eigenvectors[pivot, column] < 0.0:
            eigenvectors[:, column] *= -1.0
    if (not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all()
            or np.max(np.diff(eigenvalues)) > 1e-12 or abs(eigenvalues[0] - 1.0) > 1e-8):
        raise ValueError("invalid eigensystem")
    residual = max(float(np.linalg.norm(operator @ eigenvectors[:, j]
                                         - eigenvalues[j] * eigenvectors[:, j])) for j in range(K))
    orthogonality = float(np.max(np.abs(eigenvectors.T @ eigenvectors - np.eye(K))))
    if residual > 1e-6 or orthogonality > 1e-8:
        raise ValueError("eigensystem residual/orthogonality gate failed")
    norms = np.linalg.norm(eigenvectors, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError("invalid spectral row norm")
    embedding = eigenvectors / norms[:, None]
    if not np.isfinite(embedding).all() or np.max(np.abs(np.linalg.norm(embedding, axis=1) - 1.0)) > 2e-12:
        raise ValueError("invalid spectral embedding")
    options = dict(n_clusters=K, init="k-means++", random_state=123,
                   n_init=50, max_iter=1000, tol=1e-6, algorithm="lloyd")
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        model = KMeans(**options).fit(embedding)
        repeat = KMeans(**options).fit(embedding)
    if emitted:
        raise ValueError(f"KMeans warning: {emitted[0].message}")
    labels = np.asarray(model.labels_, dtype=np.int32)
    counts = np.bincount(labels, minlength=K)
    if (not np.array_equal(labels, repeat.labels_) or not np.isfinite(model.inertia_)
            or model.n_iter_ >= 1000 or np.any(counts == 0)):
        raise ValueError("KMeans determinism/occupation gate failed")
    return {
        "degree": degree, "operator": operator, "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors, "embedding": embedding, "labels": labels,
        "residual_max": residual, "orthogonality_max": orthogonality,
        "kmeans_inertia": float(model.inertia_), "kmeans_iterations": int(model.n_iter_),
        "cluster_min": int(counts.min()), "cluster_max": int(counts.max()),
    }


def technical_report(affinity: np.ndarray, mode: str, index_sha: str | None = None) -> dict[str, object]:
    n = affinity.shape[0]
    if not np.isfinite(affinity).all() or np.any(affinity < 0.0) or np.any(affinity > 1.0):
        raise ValueError("invalid technical affinity values")
    if not np.array_equal(affinity, affinity.T):
        raise ValueError("technical affinity must be bitwise symmetric")
    result = spectral_chain(affinity)
    w = affinity.copy()
    np.fill_diagonal(w, 0.0)
    degree_ref = w.sum(axis=1, dtype=np.float64)
    inv = 1.0 / np.sqrt(degree_ref)
    m = inv[:, None] * w * inv[None, :]
    probe = np.arange(n * 3, dtype=np.float64).reshape(n, 3) / (n * 3)
    operator_difference = float(np.max(np.abs(result["operator"] @ probe - m @ probe)))
    if (not np.array_equal(np.diag(w), np.zeros(n))
            or not np.allclose(result["degree"], degree_ref, rtol=0.0, atol=2e-14)
            or operator_difference > 1e-12):
        raise ValueError("W/degree/M operator mismatch")
    dense_values, dense_vectors = np.linalg.eigh(m)
    dense_order = np.argsort(dense_values)[::-1]
    dense_values = dense_values[dense_order]
    dense_vectors = dense_vectors[:, dense_order]
    eig_error = float(np.max(np.abs(result["eigenvalues"] - dense_values[:K])))
    projector_error = float(np.max(np.abs(
        result["eigenvectors"] @ result["eigenvectors"].T
        - dense_vectors[:, :K] @ dense_vectors[:, :K].T
    )))
    if eig_error > 1e-8 or projector_error > 1e-7:
        raise ValueError("eigsh disagrees with dense diagnostic")
    report: dict[str, object] = {
        "mode": mode, "technical_only": True, "label_free": True,
        "rows": n, "k": K, "W_diagonal_zero": True,
        "degree_min": float(np.min(result["degree"])),
        "degree_max": float(np.max(result["degree"])),
        "degree_max_abs_vs_W": float(np.max(np.abs(result["degree"] - degree_ref))),
        "M_operator_max_abs": operator_difference,
        "M_dense_diagonal_exact_zero": bool(np.array_equal(np.diag(m), np.zeros(n))),
        "eigenvalue_max_abs_vs_dense": eig_error,
        "eigenvalue_max_rel_vs_dense": float(np.max(
            np.abs(result["eigenvalues"] - dense_values[:K])
            / np.maximum(np.abs(dense_values[:K]), 1e-8))),
        "eigenspace_projector_max_abs_vs_dense": projector_error,
        "residual_max": result["residual_max"],
        "orthogonality_max": result["orthogonality_max"],
        "row_norm_error_max": float(np.max(np.abs(np.linalg.norm(result["embedding"], axis=1) - 1.0))),
        "kmeans_inertia": result["kmeans_inertia"],
        "kmeans_iterations": result["kmeans_iterations"],
        "kmeans_repeat_labels_identical": True,
        "cluster_min": result["cluster_min"], "cluster_max": result["cluster_max"],
        "eigenvalues_sha256": array_sha256(result["eigenvalues"], "<f8"),
        "eigenvectors_sha256": array_sha256(result["eigenvectors"], "<f8"),
        "embedding_sha256": array_sha256(result["embedding"], "<f8"),
        "labels_sha256": array_sha256(result["labels"], "<i4"),
        "output_created": False,
    }
    if index_sha is not None:
        report["subset_index_sha256"] = index_sha
    if mode == "technical-golden":
        gaps = np.abs(np.diff(dense_values[:K]))
        boundary_gap = float(dense_values[K - 1] - dense_values[K])
        if float(np.min(gaps)) <= 1e-6 or boundary_gap <= 1e-6:
            raise ValueError("synthetic golden eigenspectrum is degenerate")
        dense_vectors = dense_vectors[:, :K].copy()
        for column in range(K):
            pivot = int(np.argmax(np.abs(dense_vectors[:, column])))
            if dense_vectors[pivot, column] < 0.0:
                dense_vectors[:, column] *= -1.0
        vector_error = np.abs(result["eigenvectors"] - dense_vectors)
        dense_embedding = dense_vectors / np.linalg.norm(dense_vectors, axis=1)[:, None]
        embedding_error = np.abs(result["embedding"] - dense_embedding)
        report.update({
            "top_eigengap_min": float(np.min(gaps)), "boundary_eigengap": boundary_gap,
            "eigenvector_max_abs_vs_dense": float(np.max(vector_error)),
            "eigenvector_max_rel_vs_dense": float(np.max(vector_error / np.maximum(np.abs(dense_vectors), 1e-8))),
            "embedding_max_abs_vs_dense": float(np.max(embedding_error)),
            "embedding_max_rel_vs_dense": float(np.max(embedding_error / np.maximum(np.abs(dense_embedding), 1e-8))),
        })
        if report["eigenvector_max_abs_vs_dense"] > 1e-8 or report["embedding_max_abs_vs_dense"] > 1e-8:
            raise ValueError("synthetic golden vector/embedding mismatch")
    return report


def physical_memory() -> tuple[int, int]:
    if sys.platform != "win32":
        raise ValueError("Windows physical RAM preflight required")
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in (
                "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual",
            )
        ]
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def refuse_output_target(path: Path) -> None:
    path = Path(path)
    if any(Path(str(path) + suffix).exists() for suffix in ("", ".partial", ".tmp")):
        raise ValueError("output, .partial, or .tmp already exists; refusing overwrite")


def preflight_full(affinity_dir: Path, notebook_path: Path, output_dir: Path | None) -> dict[str, object]:
    validate_inputs(affinity_dir, notebook_path, full_scan=True)
    total, available = physical_memory()
    disk_target = Path(output_dir).parent if output_dir is not None else Path(affinity_dir)
    while not disk_target.exists():
        disk_target = disk_target.parent
    free_disk = shutil.disk_usage(disk_target).free
    collision = False
    output_path_valid = (output_dir is not None
                         and Path(output_dir).parent.resolve() == Path(affinity_dir).parent.resolve())
    if output_dir is not None:
        try:
            refuse_output_target(output_dir)
        except ValueError:
            collision = True
    return {
        "mode": "preflight-full", "environment": verify_environment(),
        "input_affinity_sha256": AFFINITY_SHA, "input_rows_sha256": ROWS_SHA,
        "input_affinity_spec_sha256": SPEC_SHA, "source_H_sha256": SOURCE_H_SHA,
        "iic_final_sha256": IIC_FINAL_SHA, "notebook06_sha256": NOTEBOOK_SHA,
        "affinity_bytes": N * N * 8,
        "total_physical_ram_bytes": total, "available_ram_bytes": available,
        "free_disk_bytes": free_disk, "estimated_peak_ram_bytes": PEAK_RAM_BUDGET,
        "estimated_temporary_M_disk_bytes": TEMP_M_BYTES,
        "estimated_temporary_disk_bytes": ESTIMATED_STAGING_DISK_BYTES,
        "minimum_available_ram_bytes": MIN_AVAILABLE_RAM,
        "minimum_free_disk_bytes": MIN_FREE_DISK,
        "output_collision": collision, "output_path_valid": output_path_valid,
        "full_ready": (available >= MIN_AVAILABLE_RAM and free_disk >= MIN_FREE_DISK
                       and not collision and output_path_valid),
        "eigsh_calls": 0, "kmeans_fits": 0, "output_created": False,
    }


def producer_commit() -> str:
    if subprocess.check_output(["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=all"],
                               cwd=REPO_ROOT, text=True).strip():
        raise ValueError("producer repository must be clean")
    commit = subprocess.check_output(["git", "--no-optional-locks", "rev-parse", "HEAD"],
                                     cwd=REPO_ROOT, text=True).strip()
    relative = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    committed = subprocess.check_output(["git", "--no-optional-locks", "show", f"HEAD:{relative}"], cwd=REPO_ROOT)
    if hashlib.sha256(committed).hexdigest() != sha256_file(Path(__file__)):
        raise ValueError("producer script is not byte-identical to HEAD")
    return commit


def publish_full(output_dir: Path, affinity_dir: Path, result: dict[str, object], versions: dict[str, str]) -> None:
    output_dir = Path(output_dir)
    refuse_output_target(output_dir)
    if output_dir.parent.resolve() != Path(affinity_dir).parent.resolve():
        raise ValueError("full output must be sibling of canonical affinity")
    commit = producer_commit()
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
    shutil.copyfile(Path(affinity_dir) / "rows.csv", partial / "rows.csv")
    environment = {
        **versions, "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(), "threadpools": threadpoolctl.threadpool_info(),
    }
    (partial / "environment.txt").write_text(json.dumps(environment, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    spec = {
        "format": "VAE_C_IIC_SPECTRAL_ARTIFACT_V1", "rows": N, "components": K,
        "clusters": K, "dtype_eigensystem": "float64_le", "dtype_labels": "int32_le",
        "layout": "row_major", "affinity_sha256": AFFINITY_SHA, "affinity_rows_sha256": ROWS_SHA,
        "notebook06_sha256": NOTEBOOK_SHA, "source_H_sha256": SOURCE_H_SHA,
        "iic_final_sha256": IIC_FINAL_SHA, "iic_run_id": RUN_ID,
        "eigensolver": "scipy.sparse.linalg.eigsh", "eigensolver_k": K,
        "eigensolver_which": "LA", "eigensolver_tol": 1e-8,
        "eigensolver_maxiter": 100000, "eigensolver_ncv": NCV,
        "v0": "default_rng(123).standard_normal(n).astype(float64); divide_by_l2_norm",
        "eigenvalue_order": "np.argsort(values)[::-1]", "sign_fix": "first_np_argmax_abs_positive",
        "row_normalization": "L2", "kmeans_n_clusters": K,
        "kmeans_init": "k-means++", "kmeans_seed": 123, "kmeans_n_init": 50,
        "kmeans_max_iter": 1000, "kmeans_tol": 1e-6, "kmeans_algorithm": "lloyd",
        "kmeans_inertia": result["kmeans_inertia"], "kmeans_iterations": result["kmeans_iterations"],
        "residual_max": result["residual_max"], "orthogonality_max": result["orthogonality_max"],
        "label_free": True, "producer_commit": commit,
        "producer_script_sha256": sha256_file(Path(__file__)), "environment": versions,
        "temporary_M_created": False,
    }
    (partial / "spectral_spec.txt").write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    names = sorted(path.name for path in partial.iterdir() if path.is_file())
    manifest = "".join(f"{sha256_file(partial / name)}  {name}\n" for name in names)
    (partial / "sha256.txt").write_text(manifest, encoding="ascii")
    expected = set(arrays) | {"rows.csv", "environment.txt", "spectral_spec.txt", "sha256.txt"}
    if {path.name for path in partial.iterdir()} != expected:
        raise ValueError("published file inventory mismatch")
    for line in manifest.splitlines():
        sha, name = line.split("  ", 1)
        if sha256_file(partial / name) != sha:
            raise ValueError(f"publish checksum mismatch: {name}")
    partial.rename(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("technical-golden", "technical-subset", "preflight-full", "run-full"))
    parser.add_argument("--affinity-dir", type=Path, default=DEFAULT_AFFINITY_DIR)
    parser.add_argument("--notebook-path", type=Path, default=REFERENCE_NOTEBOOK)
    parser.add_argument("--subset-size", type=int, default=256)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-full-run", action="store_true")
    args = parser.parse_args()
    versions = verify_environment()
    verify_file(args.notebook_path, NOTEBOOK_SIZE, NOTEBOOK_SHA)
    if args.mode == "technical-golden":
        rng = np.random.default_rng(20260917)
        h = rng.standard_normal((64, 5))
        affinity = np.exp(-np.linalg.norm(h[:, None, :] - h[None, :, :], axis=2))
        report = technical_report(affinity, args.mode)
    elif args.mode == "technical-subset":
        if args.subset_size < K + 2 or args.subset_size > 512:
            raise ValueError("technical subset size must be 38..512")
        canonical = validate_inputs(args.affinity_dir, args.notebook_path, full_scan=False)
        indices = np.linspace(0, N - 1, args.subset_size, dtype=np.int64)
        affinity = np.asarray(canonical[np.ix_(indices, indices)])
        report = technical_report(affinity, args.mode, array_sha256(indices, "<i8"))
    elif args.mode == "preflight-full":
        report = preflight_full(args.affinity_dir, args.notebook_path, args.output_dir)
    else:
        if not args.confirm_full_run or args.output_dir is None:
            raise ValueError("full run requires --confirm-full-run and --output-dir")
        refuse_output_target(args.output_dir)
        if args.output_dir.parent.resolve() != args.affinity_dir.parent.resolve():
            raise ValueError("full output must be sibling of canonical affinity")
        producer_commit()  # Fail before eigsh if the producer is not the committed source.
        preflight = preflight_full(args.affinity_dir, args.notebook_path, args.output_dir)
        if not preflight["full_ready"]:
            raise ValueError("full resource/input/output preflight not ready")
        canonical = np.memmap(args.affinity_dir / "affinity_f64.bin", dtype="<f8", mode="r", shape=(N, N))
        result = spectral_chain(canonical)
        publish_full(args.output_dir, args.affinity_dir, result, versions)
        report = {"mode": "run-full", "output_created": True, "output_dir": str(args.output_dir)}
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"SPECTRAL_BRIDGE_ERROR={error}", file=sys.stderr)
        sys.exit(2)
