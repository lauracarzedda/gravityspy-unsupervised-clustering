#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
)

LABELS = [
    "1080Lines", "1400Ripples", "Air_Compressor", "Blip", "Chirp",
    "Extremely_Loud", "Helix", "Koi_Fish", "Light_Modulation",
    "Low_Frequency_Burst", "Low_Frequency_Lines", "No_Glitch",
    "None_of_the_Above", "Paired_Doves", "Power_Line", "Repeating_Blips",
    "Scattered_Light", "Scratchy", "Tomte", "Violin_Mode",
    "Wandering_Line", "Whistle",
]

MINI = {
    "mini75_primary": "mini75_iic_v1_737da555866b822e14c545c4f2687869f41a52cd1340b9e941f3a1ce77763eb9_spectral_primary_v1",
    "mini112_primary": "mini112_iic_v1_0cecf5cd656d0d61a237d50fcf96a7890fb2e6d052ef403051bc9bc17aed8fba_spectral_primary_v1",
    "mini224_1000_primary": "mini224_iic_v1_4236df3f9e82f0528fcea947b2e15e42fab8159c0d4b1ed03189e4a85daae8a5_spectral_primary_v1",
}


def manifest(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return rows, {name: i for i, name in enumerate(LABELS)}


def inputs(args, manifest_rows, label_to_index):
    if args.profile in MINI:
        directory = args.repository_root / "sakai_multires_c/results/vae_c/iic" / MINI[args.profile]
        clusters = np.fromfile(directory / "cluster_labels_i32.bin", dtype="<i4")
        with (directory / "rows.csv").open(newline="", encoding="utf-8") as handle:
            row_map = list(csv.DictReader(handle))
        assert len(row_map) == clusters.size
        labels = []
        seen = set()
        for row in row_map:
            mi = int(row["manifest_index"])
            assert mi not in seen
            seen.add(mi)
            event = manifest_rows[mi]
            assert int(row["event_index"]) == int(event["event_index"])
            assert row["event_id"] == event["event_id"]
            labels.append(label_to_index[event["label"]])
        return np.asarray(labels, dtype=np.int64), clusters.astype(np.int64)
    role = "PRIMARY" if args.profile == "full224_primary" else "SENSITIVITY"
    clusters = np.load(args.full224_root / role / "partition.npy", allow_pickle=False)
    labels = np.asarray([label_to_index[row["label"]] for row in manifest_rows], dtype=np.int64)
    return labels, clusters.astype(np.int64, copy=False)


def load_c_output(directory: Path):
    with (directory / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        metric_row = next(csv.DictReader(handle))
    with (directory / "contingency_22x36.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    contingency = np.asarray(
        [[int(row[f"cluster_{cluster}"]) for cluster in range(36)] for row in rows],
        dtype=np.int64,
    )
    return metric_row, contingency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--full224-root", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_rows, label_to_index = manifest(args.repository_root / "manifest_events_272.csv")
    labels, clusters = inputs(args, manifest_rows, label_to_index)
    assert labels.shape == clusters.shape
    contingency = np.zeros((22, 36), dtype=np.int64)
    np.add.at(contingency, (labels, clusters), 1)
    majority = contingency.max(axis=0).sum() / labels.size
    reference = {
        "purity": float(majority),
        "nmi_arithmetic": normalized_mutual_info_score(labels, clusters, average_method="arithmetic"),
        "ari": adjusted_rand_score(labels, clusters),
        "homogeneity": homogeneity_score(labels, clusters),
        "completeness": completeness_score(labels, clusters),
    }
    c_metrics, c_contingency = load_c_output(args.evaluation_dir)
    assert np.array_equal(contingency, c_contingency)
    errors = {key: abs(float(c_metrics[key]) - value) for key, value in reference.items()}
    maximum = max(errors.values())
    assert maximum <= 2.0e-12, (errors, reference, c_metrics)
    print(f"PROFILE={args.profile}")
    print(f"ROWS={labels.size}")
    print(f"CONTINGENCY_SUM={int(contingency.sum())}")
    print(f"SKLEARN_MAX_ABS_ERROR={maximum:.17g}")
    print("SKLEARN_CROSSCHECK_PASS=true")


if __name__ == "__main__":
    main()
