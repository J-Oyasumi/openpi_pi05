#!/usr/bin/env bash
# One-shot install for the merged `realworld` xArm7 dataset bundle.
#
# Assumptions:
#   - realworld_dataset.zip sits in the openpi repo root (or pass --zip <path>).
#   - The script resolves the openpi repo root by walking up two levels from
#     its own location.
#
# What it does:
#   1) unzip realworld_dataset.zip to a tmp dir
#   2) move the LeRobot dataset under $HF_LEROBOT_HOME/hanjiang/realworld
#   3) move the precomputed norm_stats into <repo>/assets/pi05_realworld/...
#                                       and <repo>/assets/debug_pi05_realworld/...
#
# Training is intentionally not started here -- launch it manually afterwards, e.g.
#   uv run python scripts/train.py pi05_realworld --exp-name=run0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ZIP_PATH="${REPO_ROOT}/realworld_dataset.zip"
HF_LEROBOT_HOME_DEFAULT="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  --zip PATH      Path to realworld_dataset.zip (default: \$REPO/realworld_dataset.zip)
  --hf-home DIR   Override HF_LEROBOT_HOME (default: \$HF_LEROBOT_HOME or ~/.cache/huggingface/lerobot)
  -h, --help      Show this message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --zip) ZIP_PATH="$2"; shift 2 ;;
        --hf-home) HF_LEROBOT_HOME_DEFAULT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

cyan()   { printf "\033[1;36m%s\033[0m\n" "$*"; }
green()  { printf "\033[1;32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[1;33m%s\033[0m\n" "$*"; }
red()    { printf "\033[1;31m%s\033[0m\n" "$*"; }

cyan "[1/3] repo_root=${REPO_ROOT}"
cyan "      zip=${ZIP_PATH}"
cyan "      hf_home=${HF_LEROBOT_HOME_DEFAULT}"

if [[ ! -f "${ZIP_PATH}" ]]; then
    red "zip not found: ${ZIP_PATH}"
    exit 1
fi

STAGE_DIR="$(mktemp -d -t realworld.XXXXXX)"
trap 'rm -rf "${STAGE_DIR}"' EXIT

cyan "[2/3] Unzipping to ${STAGE_DIR}"
unzip -q "${ZIP_PATH}" -d "${STAGE_DIR}"

PKG_DIR="${STAGE_DIR}/_pkg_realworld"
if [[ ! -d "${PKG_DIR}" ]]; then
    red "Unexpected zip layout, missing ${PKG_DIR}"
    ls -R "${STAGE_DIR}" | head -20
    exit 1
fi

# 2a) LeRobot dataset -> $HF_LEROBOT_HOME/hanjiang/realworld
DST_DATA="${HF_LEROBOT_HOME_DEFAULT}/hanjiang/realworld"
cyan "[2/3] Installing dataset -> ${DST_DATA}"
mkdir -p "${HF_LEROBOT_HOME_DEFAULT}/hanjiang"
if [[ -e "${DST_DATA}" ]]; then
    yellow "      removing existing ${DST_DATA}"
    rm -rf "${DST_DATA}"
fi
mv "${PKG_DIR}/lerobot_dataset/hanjiang/realworld" "${DST_DATA}"

# 2b) norm_stats -> <repo>/assets/<config>/hanjiang/realworld
install_assets() {
    local src_dir="$1"
    local dst_cfg="$2"
    local dst_dir="${REPO_ROOT}/assets/${dst_cfg}/hanjiang/realworld"
    if [[ -d "${src_dir}" ]]; then
        mkdir -p "$(dirname "${dst_dir}")"
        rm -rf "${dst_dir}"
        mv "${src_dir}/hanjiang/realworld" "${dst_dir}"
        green "      installed norm_stats -> ${dst_dir}"
    else
        yellow "      ${src_dir} not in zip, skipping ${dst_cfg}"
    fi
}
cyan "[3/3] Installing norm_stats"
install_assets "${PKG_DIR}/assets_pi05"  "pi05_realworld"
install_assets "${PKG_DIR}/assets_debug" "debug_pi05_realworld"

green "[done] realworld dataset & norm_stats installed"
echo
echo "Next:  uv run python scripts/train.py pi05_realworld --exp-name=run0"
