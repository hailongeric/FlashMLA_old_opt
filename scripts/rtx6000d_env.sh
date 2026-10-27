#!/usr/bin/env bash
# Source this file; all generated state stays in the authorized project tree.
_mla_root="${HOME}/nhl_project"
source "${_mla_root}/nhlpy3/bin/activate"
export PIP_CACHE_DIR="${_mla_root}/.cache/pip"
export TRITON_CACHE_DIR="${_mla_root}/.cache/triton"
export CUDA_CACHE_PATH="${_mla_root}/.cache/cuda"
export TORCH_EXTENSIONS_DIR="${_mla_root}/.cache/torch_extensions"
export XDG_CACHE_HOME="${_mla_root}/.cache"
export TMPDIR="${_mla_root}/.tmp"
export PYTHONDONTWRITEBYTECODE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
# This shared host has no international package-index access.
export PIP_NO_INDEX=1
_mla_headers="${_mla_root}/toolchains/python3.12-headers/usr/include"
if [[ -f "${_mla_headers}/python3.12/Python.h" ]]; then
    export CPATH="${_mla_headers}/python3.12:${_mla_headers}${CPATH:+:${CPATH}}"
fi
mkdir -p "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR" "$TMPDIR"
unset _mla_root _mla_headers
