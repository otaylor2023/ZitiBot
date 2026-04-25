#!/usr/bin/env bash
# Start simviz, then start controller. Use --wait to require SPACE before the controller.
# Controller runs in the background so Ctrl+C hits this shell and stops both processes.
set -euo pipefail

WAIT_FOR_SPACE=false
RUN_BUILD=false
for arg in "$@"; do
	case "${arg}" in
		--wait) WAIT_FOR_SPACE=true ;;
		--build) RUN_BUILD=true ;;
		-h | --help)
			echo "Usage: $(basename "$0") [--wait] [--build]"
			echo "  default   visualizer, countdown, controller (no build)"
			echo "  --wait    wait for SPACE in this terminal before starting the controller"
			echo "  --build   run CMake configure + build for ZitiBot targets, then launch"
			echo "Env: JOBS, CMAKE_BUILD_TYPE (same as build_zitibot.sh)"
			echo "Redis must already be running."
			exit 0
			;;
		*)
			echo "Unknown option: ${arg}  (try --help)"
			exit 1
			;;
	esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BIN_DIR="${SCRIPT_DIR}/bin/zitibot_mmp_example"
SIMVIZ="${BIN_DIR}/simviz_zitibot_mmp_panda"
CTRL="${BIN_DIR}/controller_zitibot_mmp_panda"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

if [[ "${RUN_BUILD}" == true ]]; then
	mkdir -p "${BUILD_DIR}"
	echo "==> Configuring (${BUILD_DIR})"
	cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
	echo "==> Building ZitiBot targets (${JOBS} jobs)"
	cmake --build "${BUILD_DIR}" -j "${JOBS}" --target controller_zitibot_mmp_panda simviz_zitibot_mmp_panda
fi

for exe in "${SIMVIZ}" "${CTRL}"; do
	if [[ ! -x "${exe}" ]]; then
		echo "Missing executable: ${exe}" >&2
		echo "Run: ${SCRIPT_DIR}/build_zitibot.sh   or   ${SCRIPT_DIR}/launch_zitibot.sh --build" >&2
		exit 1
	fi
done

SIMVIZ_PID=""
CTRL_PID=""

cleanup() {
	if [[ -n "${CTRL_PID}" ]] && kill -0 "${CTRL_PID}" 2>/dev/null; then
		kill -TERM "${CTRL_PID}" 2>/dev/null || true
		wait "${CTRL_PID}" 2>/dev/null || true
	fi
	if [[ -n "${SIMVIZ_PID}" ]] && kill -0 "${SIMVIZ_PID}" 2>/dev/null; then
		kill -TERM "${SIMVIZ_PID}" 2>/dev/null || true
		wait "${SIMVIZ_PID}" 2>/dev/null || true
	fi
}

on_interrupt() {
	echo "" >&2
	echo "Interrupted — stopping controller and visualizer." >&2
	cleanup
	trap - INT TERM EXIT
	exit 130
}

trap cleanup EXIT
trap on_interrupt INT TERM

echo "Starting visualizer..."
"${SIMVIZ}" &
SIMVIZ_PID=$!

echo -n "Waiting for visualizer"
deadline=$((SECONDS + 15))
while (( SECONDS < deadline )); do
	if kill -0 "${SIMVIZ_PID}" 2>/dev/null; then
		echo " — ok (pid ${SIMVIZ_PID})"
		break
	fi
	echo -n "."
	sleep 0.2
done

if ! kill -0 "${SIMVIZ_PID}" 2>/dev/null; then
	echo ""
	echo "Visualizer exited before it could start. Check Redis and try again."
	exit 1
fi

if [[ "${WAIT_FOR_SPACE}" == true ]]; then
	echo ""
	echo "Visualizer is running. Press SPACE in this terminal to start the controller."
	echo "(Ctrl+C stops the visualizer.)"
	echo ""
	while IFS= read -r -n1 key; do
		if [[ "${key}" == " " ]]; then
			break
		fi
	done
else
	sleep 0.5
	echo ""
	echo "Starting controller in:"
	for i in 3 2 1; do
		echo "  ${i}..."
		sleep 1
	done
	echo "  Go!"
	echo ""
fi

echo "Starting controller (Ctrl+C here stops both; Q in the sim window quits visualizer and this script)..."
"${CTRL}" &
CTRL_PID=$!

# Exit when either process dies (e.g. Q in sim closes visualizer → we stop controller and the script ends).
while kill -0 "${SIMVIZ_PID}" 2>/dev/null && kill -0 "${CTRL_PID}" 2>/dev/null; do
	sleep 0.15
done
if ! kill -0 "${SIMVIZ_PID}" 2>/dev/null && kill -0 "${CTRL_PID}" 2>/dev/null; then
	echo "Visualizer exited (e.g. Q); stopping controller." >&2
fi
if ! kill -0 "${CTRL_PID}" 2>/dev/null && kill -0 "${SIMVIZ_PID}" 2>/dev/null; then
	echo "Controller exited; stopping visualizer." >&2
fi
# EXIT trap runs cleanup for anything still alive
