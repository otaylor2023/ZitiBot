#!/usr/bin/env bash
# Build ZitiBot from this repository (standalone CMake project).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

mkdir -p "${BUILD_DIR}"
echo "==> Configuring ZitiBot (source: ${SCRIPT_DIR})"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"

if [[ "${1:-}" == "--all" ]]; then
	echo "==> Building all targets (${JOBS} jobs)"
	cmake --build "${BUILD_DIR}" -j "${JOBS}"
else
	echo "==> Building ZitiBot MMP targets (${JOBS} jobs)"
	cmake --build "${BUILD_DIR}" -j "${JOBS}" --target controller_zitibot_mmp_panda simviz_zitibot_mmp_panda
fi

BIN="${SCRIPT_DIR}/bin/zitibot_mmp_example"
echo ""
echo "Done. Binaries:"
echo "  ${BIN}/simviz_zitibot_mmp_panda"
echo "  ${BIN}/controller_zitibot_mmp_panda"
