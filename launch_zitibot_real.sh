#!/usr/bin/env bash
# Real robot: start only the ZitiBot controller (no simviz, no simulation).
#
# How this fits together (same idea as README "simulation and physical robot"):
#   1) Redis must be running (usually localhost:6379).
#   2) A *hardware / bridge* process must already be running on the robot PC: it should
#      publish joint state and consume torques using the keys in
#      zitibot_mmp_panda/redis_keys.h (Franka sensors + control_torques). Your
#      course/lab stack provides this driver; it is not built from this ZitiBot repo.
#   3) By default this script runs the **touch** controller (simple EE goal demo).
#      Use --main for the full pick FSM (controller_zitibot_mmp_panda), which needs
#      BOWL_POSITION_KEY (or your fork). The touch binary does not need the bowl key.
#   4) Clear the workspace, enable the robot, then run this script.
#
# Controller runs in the background; Ctrl+C in this terminal stops it.
set -euo pipefail

WAIT_FOR_SPACE=false
RUN_BUILD=false
TOUCH_DEMO=true
GEMINI_TOUCH=false
for arg in "$@"; do
	case "${arg}" in
		--wait) WAIT_FOR_SPACE=true ;;
		--build) RUN_BUILD=true ;;
		--main) TOUCH_DEMO=false ;;
		--touch) TOUCH_DEMO=true ;;
		--gemini)
			GEMINI_TOUCH=true
			TOUCH_DEMO=true
			;;
		-h | --help)
			echo "Usage: $(basename "$0") [--wait] [--build] [--main] [--touch] [--gemini]"
			echo "  Real robot: starts only a controller (no simviz)."
			echo "  Default: controller_touch_zitibot_mmp_panda (touch / offset demo)."
			echo "  --main    run controller_zitibot_mmp_panda (full pick FSM; needs bowl pose in Redis)"
			echo "  --touch   force touch controller (same as default)"
			echo "  --wait    require SPACE in this terminal before starting the controller"
			echo "  --build   CMake configure + build controller binaries only (no simviz)"
			echo "  --gemini  touch controller with --gemini (follow gemini_target_ee_* on Redis)"
			echo ""
			echo "Prerequisites:"
			echo "  - Redis running."
			echo "  - Lab hardware/Redis bridge running (joint sensors -> Redis; torques <- Redis)."
			echo "  - Robot clear and enabled; you understand the motion that will run."
			echo "Env: JOBS, CMAKE_BUILD_TYPE (same as build_zitibot.sh)"
			exit 0
			;;
		*)
			echo "Unknown option: ${arg}  (try --help)" >&2
			exit 1
			;;
	esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BIN_DIR="${SCRIPT_DIR}/bin/zitibot_mmp_example"
CTRL="${BIN_DIR}/controller_zitibot_mmp_panda"
CTRL_TOUCH="${BIN_DIR}/controller_touch_zitibot_mmp_panda"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

if [[ "${RUN_BUILD}" == true ]]; then
	mkdir -p "${BUILD_DIR}"
	echo "==> Configuring (${BUILD_DIR})"
	cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
	echo "==> Building controllers only (${JOBS} jobs)"
	cmake --build "${BUILD_DIR}" -j "${JOBS}" \
		--target controller_zitibot_mmp_panda controller_touch_zitibot_mmp_panda
fi

CTRL_EXE="${CTRL_TOUCH}"
if [[ "${TOUCH_DEMO}" == false ]]; then
	CTRL_EXE="${CTRL}"
fi

if [[ ! -x "${CTRL_EXE}" ]]; then
	echo "Missing executable: ${CTRL_EXE}" >&2
	echo "Run: ${SCRIPT_DIR}/build_zitibot.sh   or   ${SCRIPT_DIR}/launch_zitibot_real.sh --build" >&2
	exit 1
fi

if command -v redis-cli >/dev/null 2>&1; then
	if ! redis-cli ping >/dev/null 2>&1; then
		echo "Warning: redis-cli ping failed — is Redis up on this machine?" >&2
	fi
else
	echo "(Install redis-cli to get an automatic Redis ping check.)"
fi

CTRL_PID=""

cleanup() {
	if [[ -n "${CTRL_PID}" ]] && kill -0 "${CTRL_PID}" 2>/dev/null; then
		kill -TERM "${CTRL_PID}" 2>/dev/null || true
		wait "${CTRL_PID}" 2>/dev/null || true
	fi
}

on_interrupt() {
	echo "" >&2
	echo "Interrupted — stopping controller." >&2
	cleanup
	trap - INT TERM EXIT
	exit 130
}

trap cleanup EXIT
trap on_interrupt INT TERM

echo ""
echo "=== ZitiBot REAL controller ==="
echo "Not starting simviz. Ensure your robot Redis bridge is running and the arm is safe to move."
if [[ "${TOUCH_DEMO}" == true ]]; then
	echo "Controller: touch demo (default). Use --main for full FSM."
else
	echo "Controller: main pick FSM (--main)."
fi
echo ""

if [[ "${WAIT_FOR_SPACE}" == true ]]; then
	echo "Press SPACE in this terminal to start the controller. (Ctrl+C aborts.)"
	echo ""
	while IFS= read -r -n1 key; do
		if [[ "${key}" == " " ]]; then
			break
		fi
	done
else
	echo "Starting controller in:"
	for i in 3 2 1; do
		echo "  ${i}..."
		sleep 1
	done
	echo "  Go!"
	echo ""
fi

echo "Starting: ${CTRL_EXE}"
if [[ "${GEMINI_TOUCH}" == true ]]; then
	echo "(touch controller with --gemini: uses vision Redis goal when active)"
fi
echo "Ctrl+C here stops the controller. 'q' + Enter also stops (while this script waits)."
if [[ "${GEMINI_TOUCH}" == true ]]; then
	"${CTRL_EXE}" --gemini &
else
	"${CTRL_EXE}" &
fi
CTRL_PID=$!

while kill -0 "${CTRL_PID}" 2>/dev/null; do
	if IFS= read -r -n1 -t 1 key; then
		if [[ "${key}" == "q" || "${key}" == "Q" ]]; then
			echo "Quit key pressed; stopping controller." >&2
			break
		fi
	fi
done

# EXIT trap runs cleanup
