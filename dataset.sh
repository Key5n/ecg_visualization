#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

set -a
source ".env"
set +a

rm -rf "$DATASET_ROOT"
mkdir -p "$DATASET_ROOT"

download_dataset() {
  local url="$1"
  wget -P "$DATASET_ROOT" -r -N -c -np "$url"
}

# PAF Prediction Challenge Database
download_dataset "https://physionet.org/files/afpdb/1.0.0/"
# CU Ventricular Tachyarrrhythmia Database
download_dataset "https://physionet.org/files/cudb/1.0.0/"
# MIT-BIH Arrythmia Database
download_dataset "https://physionet.org/files/mitdb/1.0.0/"
# MIT-BIH Atrial Fibrillation Database
download_dataset "https://physionet.org/files/afdb/1.0.0/"
# Long Term AF Database
download_dataset "https://physionet.org/files/ltafdb/1.0.0/"
# SHDB-AF: a Japanese Holter ECG database of atrial fibrillation
download_dataset "https://physionet.org/files/shdb-af/1.0.1/"
# Sudden Cardiac Death Holter Database
download_dataset "https://physionet.org/files/sddb/1.0.0/"
# MIT-BIH Malignant Ventricular Ectopy Database
download_dataset "https://physionet.org/files/vfdb/1.0.0/"
