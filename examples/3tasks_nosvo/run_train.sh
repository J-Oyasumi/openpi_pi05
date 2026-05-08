#!/usr/bin/env bash
# One-shot setup + training for the three_tasks_nosvo dataset.
#
# Assumptions:
#   - You are running this from the openpi repo root (or via its absolute path).
#   - three_tasks_nosvo_dataset.zip is in the repo root (or pass --zip <path>).
#   - The code changes (config.py / three_tasks_nosvo_policy.py / examples/3tasks_nosvo/)
#     are already present in the checked-out repo.
#   - `uv` is installed.
#
# What it does:
#   1) unzip three_tasks_nosvo_dataset.zip
#   2) move the LeRobot dataset under $HF_LEROBOT_HOME
#   3) move the precomputed norm_stats into <repo>/assets/<config_name>/...
#   4) launch training with the chosen config + exp_name + flags

set -euo pipefail

# ---------- defaults ----------
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ZIP_PATH="${REPO_ROOT}/three_tasks_nosvo_dataset.zip"
CONFIG_NAME="pi05_three_tasks_nosvo"      # debug_pi05_three_tasks_nosvo for dry-run
EXP_NAME="run0"
EXTRA_ARGS=()
SKIP_UNPACK=0
JAX_PLATFORM=""                            # leave empty for default GPU; "cpu" for CPU dry-run
HF_LEROBOT_HOME_DEFAULT="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"

usage() {
    cat <<EOF
Usage: $0 [options] [-- <extra args forwarded to scripts/train.py>]

Options:
  --zip PATH               Path to three_tasks_nosvo_dataset.zip (default: \$REPO/three_tasks_nosvo_dataset.zip)
  --config NAME            Train config name (default: $CONFIG_NAME)
  --exp-name NAME          Experiment name (default: $EXP_NAME)
  --skip-unpack            Skip unzip + dataset/asset move (data already in place)
  --jax-platform PLATFORM  Set JAX_PLATFORMS (e.g. "cpu" for CPU dry-run)
  --hf-home DIR            Override HF_LEROBOT_HOME target (default: \$HF_LEROBOT_HOME or ~/.cache/huggingface/lerobot)
  -h, --help               Show this message

Examples:
  # Real fine-tune on GPU
  $0 --config pi05_three_tasks_nosvo --exp-name run0

  # CPU dry-run
  $0 --config debug_pi05_three_tasks_nosvo --exp-name dryrun --jax-platform cpu

  # Forward extra flags (after a "--") to train.py
  $0 --config pi05_three_tasks_nosvo --exp-name run0 -- --batch-size=16 --fsdp-devices=4
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --zip) ZIP_PATH="$2"; shift 2 ;;
        --config) CONFIG_NAME="$2"; shift 2 ;;
        --exp-name) EXP_NAME="$2"; shift 2 ;;
        --skip-unpack) SKIP_UNPACK=1; shift ;;
        --jax-platform) JAX_PLATFORM="$2"; shift 2 ;;
        --hf-home) HF_LEROBOT_HOME_DEFAULT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

cyan() { printf "\033[1;36m%s\033[0m\n" "$*"; }
green() { printf "\033[1;32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[1;33m%s\033[0m\n" "$*"; }
red() { printf "\033[1;31m%s\033[0m\n" "$*"; }

cyan "[1/4] repo_root=${REPO_ROOT}"
cyan "      zip=${ZIP_PATH}"
cyan "      config=${CONFIG_NAME}"
cyan "      exp_name=${EXP_NAME}"
cyan "      hf_home=${HF_LEROBOT_HOME_DEFAULT}"
[[ -n "${JAX_PLATFORM}" ]] && cyan "      JAX_PLATFORMS=${JAX_PLATFORM}"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && cyan "      extra train.py args: ${EXTRA_ARGS[*]}"

# ---------- unpack ----------
if [[ "${SKIP_UNPACK}" -eq 0 ]]; then
    if [[ ! -f "${ZIP_PATH}" ]]; then
        red "[1/4] zip not found: ${ZIP_PATH}"
        exit 1
    fi

    STAGE_DIR="$(mktemp -d -t three_tasks_nosvo.XXXXXX)"
    trap 'rm -rf "${STAGE_DIR}"' EXIT
    cyan "[2/4] Unzipping to ${STAGE_DIR}"
    unzip -q "${ZIP_PATH}" -d "${STAGE_DIR}"

    PKG_DIR="${STAGE_DIR}/_pkg_three_tasks_nosvo"
    if [[ ! -d "${PKG_DIR}" ]]; then
        red "Unexpected zip layout, missing ${PKG_DIR}"
        ls -R "${STAGE_DIR}" | head -20
        exit 1
    fi

    # 2a) LeRobot dataset -> $HF_LEROBOT_HOME/hanjiang/three_tasks_nosvo
    DST_DATA="${HF_LEROBOT_HOME_DEFAULT}/hanjiang/three_tasks_nosvo"
    cyan "[2/4] Installing dataset -> ${DST_DATA}"
    mkdir -p "${HF_LEROBOT_HOME_DEFAULT}/hanjiang"
    if [[ -e "${DST_DATA}" ]]; then
        yellow "      removing existing ${DST_DATA}"
        rm -rf "${DST_DATA}"
    fi
    mv "${PKG_DIR}/lerobot_dataset/hanjiang/three_tasks_nosvo" "${DST_DATA}"

    # 2b) norm_stats -> <repo>/assets/<config>/hanjiang/three_tasks_nosvo
    install_assets() {
        local src_dir="$1"
        local dst_cfg="$2"
        local dst_dir="${REPO_ROOT}/assets/${dst_cfg}/hanjiang/three_tasks_nosvo"
        if [[ -d "${src_dir}" ]]; then
            mkdir -p "$(dirname "${dst_dir}")"
            rm -rf "${dst_dir}"
            mv "${src_dir}/hanjiang/three_tasks_nosvo" "${dst_dir}"
            green "      installed norm_stats -> ${dst_dir}"
        else
            yellow "      ${src_dir} not in zip, skipping ${dst_cfg}"
        fi
    }
    cyan "[3/4] Installing norm_stats"
    install_assets "${PKG_DIR}/assets_pi05"  "pi05_three_tasks_nosvo"
    install_assets "${PKG_DIR}/assets_debug" "debug_pi05_three_tasks_nosvo"
else
    yellow "[1-3/4] --skip-unpack: assuming dataset & assets already in place"
fi

# ---------- export HF_LEROBOT_HOME for the training run ----------
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME_DEFAULT}"

# ---------- launch training ----------
cyan "[4/4] Launching training: config=${CONFIG_NAME} exp_name=${EXP_NAME}"

CMD=(uv run python "${REPO_ROOT}/scripts/train.py" "${CONFIG_NAME}" --exp-name="${EXP_NAME}")
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

if [[ -n "${JAX_PLATFORM}" ]]; then
    cyan "      JAX_PLATFORMS=${JAX_PLATFORM} ${CMD[*]}"
    JAX_PLATFORMS="${JAX_PLATFORM}" "${CMD[@]}"
else
    cyan "      ${CMD[*]}"
    "${CMD[@]}"
fi

green "[done] training finished"
