#!/usr/bin/env bash
# Downloads the fMoW-Sentinel VALIDATION split only (images + val.csv) from
# the Stanford Digital Repository (DOI 10.25740/vg497cb6002,
# https://purl.stanford.edu/vg497cb6002).
#
# NOTE: the SDR item does not expose a single stable direct-download URL for
# the "val" subset and csv in the way Zenodo does -- you must browse the
# item page, accept the fMoW-Sentinel / Sentinel-2 usage terms, and copy the
# actual file link(s) for the validation split before running this script.
# Replace the TODO URLs below once you have them.
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
VAL_IMAGES_URL="TODO_REPLACE_WITH_STANFORD_SDR_VAL_SPLIT_URL"   # e.g. fmow-sentinel val.tar.gz / val/ prefix
VAL_CSV_URL="TODO_REPLACE_WITH_STANFORD_SDR_VAL_CSV_URL"        # val.csv

mkdir -p "${DATA_DIR}/fmow-sentinel/val"

if [[ "${VAL_CSV_URL}" == TODO_* || "${VAL_IMAGES_URL}" == TODO_* ]]; then
  cat <<'EOF'
ERROR: download_data.sh has not been configured with real URLs yet.

1. Open https://purl.stanford.edu/vg497cb6002 in a browser.
2. Accept the fMoW-Sentinel / Sentinel-2 data terms.
3. Find the validation-split archive and val.csv download links.
4. Edit this script (VAL_IMAGES_URL, VAL_CSV_URL) with those links, or
   download manually into:
     data/fmow-sentinel/val/        (imagery, category/<cat>_<loc_id>/ subfolders)
     data/val.csv                   (metadata csv)
5. Re-run this script, or skip it if you already placed the files.
EOF
  exit 1
fi

echo "Downloading val.csv..."
wget --continue -O "${DATA_DIR}/val.csv" "${VAL_CSV_URL}"

echo "Downloading validation imagery archive..."
wget --continue -O "${DATA_DIR}/fmow-sentinel-val.tar.gz" "${VAL_IMAGES_URL}"

echo "Extracting..."
tar -xzf "${DATA_DIR}/fmow-sentinel-val.tar.gz" -C "${DATA_DIR}/fmow-sentinel/val"

echo "Done. Next step: python scripts/prepare_csv.py --csv data/val.csv --data-root data --out data/val_prepared.csv"
