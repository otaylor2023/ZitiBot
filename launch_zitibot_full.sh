#!/usr/bin/env bash
# Start OpenSai (arm/cartesian), Franka gripper Redis driver, TidyBot base Redis driver,
# then a ZitiBot Python controller (e.g. pour_and_move_controller.py).
#
# Does NOT start OptiTrack / NatNet — run Motive publisher manually (tidybot01::* on Redis).
# Does NOT start sai_franka_robot_redis_driver (real arm torques / RT driver).
#
# Usage:
#   ./launch_zitibot_full.sh controllers/pour_and_move_controller.py
#   ./launch_zitibot_full.sh --wait controllers/grasp_and_pour_controller.py
#   ./launch_zitibot_full.sh controllers/pour_and_move_controller.py -- --grasp-opti-x -2.52
#   ./launch_zitibot_full.sh --no-base controllers/touch_controller.py
#   ./launch_zitibot_full.sh --no-gripper controllers/touch_controller.py
#
# Prerequisites:
#   - OpenSai built (bin/OpenSai_main)
#   - Gripper binary built unless --no-gripper (launched via launch_gripper.sh):
#       drivers/FrankaPanda/redis_driver/build/sai_franka_gripper_redis_driver
#       drivers/FrankaPanda/redis_driver/launch_gripper.sh
#   - launch_gripper.sh runs `sudo -S cpufreq-set ...` with the local password
#     baked in; if you change the user's password, update launch_gripper.sh too.
#   - Robot mini-PC / CAN for base driver unless --no-base
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_XML="config_folder/xml_config_files/zitibot_panda.xml"
LAUNCH_SH="${REPO_ROOT}/scripts/launch.sh"
OPENSAI_MAIN="${REPO_ROOT}/bin/OpenSai_main"
PYTHON="${PYTHON:-python3}"

GRIPPER_DIR="${REPO_ROOT}/drivers/FrankaPanda/redis_driver"
GRIPPER_BIN="${GRIPPER_DIR}/build/sai_franka_gripper_redis_driver"
GRIPPER_LAUNCH_SH="${GRIPPER_DIR}/launch_gripper.sh"

TIDYBOT_BASE="${SCRIPT_DIR}/controllers/tidybot_base"
BASE_PID_FILE="/tmp/tidybot2-base-controller.pid"

CONFIG_REDIS_KEY="::sai-interfaces-webui::config_file_name"
READY_TIMEOUT_SEC=90
DRIVER_STARTUP_SEC=3
DRIVER_READY_TIMEOUT_SEC=15

MAX_VEL_XY="${MAX_VEL_XY:-0.25}"
MAX_VEL_YAW="${MAX_VEL_YAW:-0.79}"
MAX_ACCEL_XY="${MAX_ACCEL_XY:-0.1}"
MAX_ACCEL_YAW="${MAX_ACCEL_YAW:-0.5}"

WAIT_FOR_SPACE=false
LAUNCH_GRIPPER=true
LAUNCH_BASE=true
TUNE_MARKER_OFFSET=false
CONTROLLER_ARG=""
CONTROLLER_ARGS=()

LAUNCH_PID=""
GRIPPER_PID=""
BASE_PID=""
GRIPPER_LOG=""
BASE_LOG=""

usage() {
	echo "Usage: $(basename "$0") [options] [--wait] <controller.py> [-- controller.py args...]"
	echo ""
	echo "Starts (in order):"
	echo "  1) scripts/launch.sh ${CONFIG_XML}  (OpenSai + Redis + web UI)"
	echo "  2) sai_franka_gripper_redis_driver  (unless --no-gripper)"
	echo "  3) tidybot_base/redis_driver.py     (unless --no-base)"
	echo "  4) Python controller (foreground, stdin for ENTER)"
	echo ""
	echo "Controller path resolution (first match wins):"
	echo "  1. exact path as given"
	echo "  2. relative to ZitiBot/"
	echo "  3. relative to ZitiBot/controllers/"
	echo ""
	echo "Options:"
	echo "  --wait              press SPACE before starting the controller"
	echo "  --tune-marker-offset"
	echo "                      run controllers/tune_marker_offset.py before controller"
	echo "  --no-gripper        skip Franka gripper Redis driver"
	echo "  --no-base           skip TidyBot base redis_driver"
	echo "  --max-vel-xy M      base planar max speed m/s (default: ${MAX_VEL_XY})"
	echo "  --max-vel-yaw R     base max yaw rate rad/s (default: ${MAX_VEL_YAW})"
	echo "  --max-accel-xy A    base planar accel m/s^2 (default: ${MAX_ACCEL_XY})"
	echo "  --max-accel-yaw A   base yaw accel rad/s^2 (default: ${MAX_ACCEL_YAW})"
	echo "  -h, --help"
	exit 0
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		-h | --help) usage ;;
		--wait) WAIT_FOR_SPACE=true; shift ;;
		--tune-marker-offset) TUNE_MARKER_OFFSET=true; shift ;;
		--no-gripper) LAUNCH_GRIPPER=false; shift ;;
		--no-base) LAUNCH_BASE=false; shift ;;
		--max-vel-xy) MAX_VEL_XY="$2"; shift 2 ;;
		--max-vel-yaw) MAX_VEL_YAW="$2"; shift 2 ;;
		--max-accel-xy) MAX_ACCEL_XY="$2"; shift 2 ;;
		--max-accel-yaw) MAX_ACCEL_YAW="$2"; shift 2 ;;
		--) shift; CONTROLLER_ARGS+=("$@"); break ;;
		-*)
			echo "Unknown option: $1" >&2
			exit 1
			;;
		*)
			if [[ -z "${CONTROLLER_ARG}" ]]; then
				CONTROLLER_ARG="$1"
				shift
			else
				CONTROLLER_ARGS+=("$1")
				shift
			fi
			;;
	esac
done

if [[ -z "${CONTROLLER_ARG}" ]]; then
	echo "Error: controller file path required." >&2
	echo "Run: $(basename "$0") --help" >&2
	exit 1
fi

resolve_controller_path() {
	local arg="$1"
	local candidates=(
		"${arg}"
		"${SCRIPT_DIR}/${arg}"
		"${SCRIPT_DIR}/controllers/${arg}"
	)
	for c in "${candidates[@]}"; do
		if [[ -f "${c}" ]]; then
			(cd "$(dirname "${c}")" && printf '%s/%s\n' "$(pwd)" "$(basename "${c}")")
			return 0
		fi
	done
	return 1
}

if ! CONTROLLER_PATH="$(resolve_controller_path "${CONTROLLER_ARG}")"; then
	echo "Error: controller file not found: ${CONTROLLER_ARG}" >&2
	echo "Looked in: cwd, ${SCRIPT_DIR}/, ${SCRIPT_DIR}/controllers/" >&2
	exit 1
fi

CONTROLLER_DIR="$(dirname "${CONTROLLER_PATH}")"
CONTROLLER_SCRIPT="$(basename "${CONTROLLER_PATH}")"
CONTROLLER_NAME="${CONTROLLER_SCRIPT%.py}"

if [[ ! -f "${LAUNCH_SH}" ]]; then
	echo "Missing ${LAUNCH_SH}" >&2
	exit 1
fi
if [[ ! -x "${OPENSAI_MAIN}" ]]; then
	echo "Missing ${OPENSAI_MAIN} — build OpenSai first." >&2
	exit 1
fi
if [[ "${LAUNCH_GRIPPER}" == true ]]; then
	if [[ ! -x "${GRIPPER_BIN}" ]]; then
		echo "Missing ${GRIPPER_BIN}" >&2
		echo "Build drivers/FrankaPanda/redis_driver or pass --no-gripper." >&2
		exit 1
	fi
	if [[ ! -f "${GRIPPER_LAUNCH_SH}" ]]; then
		echo "Missing ${GRIPPER_LAUNCH_SH}" >&2
		exit 1
	fi
fi
if [[ "${LAUNCH_BASE}" == true && ! -f "${TIDYBOT_BASE}/redis_driver.py" ]]; then
	echo "Missing ${TIDYBOT_BASE}/redis_driver.py" >&2
	exit 1
fi
if [[ "${TUNE_MARKER_OFFSET}" == true && "${LAUNCH_BASE}" != true ]]; then
	echo "Error: --tune-marker-offset requires the base driver; remove --no-base." >&2
	exit 1
fi

if command -v redis-cli >/dev/null 2>&1; then
	if ! redis-cli ping >/dev/null 2>&1; then
		echo "Warning: redis-cli ping failed — launch.sh may start redis-server." >&2
	fi
else
	echo "Warning: redis-cli not found; cannot verify Redis keys." >&2
fi

stop_tree() {
	local pid="$1"
	[[ -z "${pid}" ]] && return 0
	if ! kill -0 "${pid}" 2>/dev/null; then
		return 0
	fi
	pkill -TERM -P "${pid}" 2>/dev/null || true
	kill -TERM "${pid}" 2>/dev/null || true
	local i
	for i in 1 2 3 4 5 6 7 8 9 10; do
		kill -0 "${pid}" 2>/dev/null || return 0
		sleep 0.2
	done
	pkill -KILL -P "${pid}" 2>/dev/null || true
	kill -KILL "${pid}" 2>/dev/null || true
	wait "${pid}" 2>/dev/null || true
}

stop_process() {
	local pid="$1"
	local label="$2"
	[[ -z "${pid}" ]] && return 0
	if ! kill -0 "${pid}" 2>/dev/null; then
		return 0
	fi
	echo "Stopping ${label} (pid ${pid})..."
	kill -TERM "${pid}" 2>/dev/null || true
	local i
	for i in 1 2 3 4 5 6 7 8 9 10; do
		kill -0 "${pid}" 2>/dev/null || break
		sleep 0.2
	done
	if kill -0 "${pid}" 2>/dev/null; then
		kill -KILL "${pid}" 2>/dev/null || true
	fi
	wait "${pid}" 2>/dev/null || true
}

remove_base_pid_file() {
	rm -f "${BASE_PID_FILE}"
}

preflight_cleanup_base() {
	local old_pid=""
	if [[ "${LAUNCH_BASE}" != true ]]; then
		return 0
	fi
	if [[ -f "${BASE_PID_FILE}" ]]; then
		old_pid="$(tr -d '[:space:]' <"${BASE_PID_FILE}" 2>/dev/null || true)"
	fi
	if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
		if [[ -r "/proc/${old_pid}/status" ]] && grep -q '^State:[[:space:]]*Z' "/proc/${old_pid}/status"; then
			echo "Removing stale base-controller lock (zombie pid ${old_pid})"
			remove_base_pid_file
		else
			echo "Stopping previous redis_driver (pid ${old_pid})..."
			stop_process "${old_pid}" "previous base driver"
			remove_base_pid_file
		fi
	fi
	pkill -TERM -f "${TIDYBOT_BASE}/redis_driver.py" 2>/dev/null || true
	sleep 0.3
	pkill -KILL -f "${TIDYBOT_BASE}/redis_driver.py" 2>/dev/null || true
	remove_base_pid_file
}

show_log_tail() {
	local log="$1"
	local label="$2"
	if [[ -n "${log}" && -f "${log}" ]]; then
		echo "--- ${label} (${log}) ---" >&2
		tail -n 80 "${log}" >&2 || true
		echo "--- end log ---" >&2
	fi
}

cleanup() {
	stop_process "${BASE_PID}" "redis_driver"
	if [[ -n "${GRIPPER_PID}" ]]; then
		echo "Stopping gripper driver (pid ${GRIPPER_PID})..."
		stop_tree "${GRIPPER_PID}"
		GRIPPER_PID=""
	fi
	pkill -TERM -f "${GRIPPER_BIN}" 2>/dev/null || true
	remove_base_pid_file
	if [[ -n "${LAUNCH_PID}" ]]; then
		stop_tree "${LAUNCH_PID}"
		LAUNCH_PID=""
	fi
	tmux kill-session -t interfaces_server 2>/dev/null || true
	pkill -TERM -f "${OPENSAI_MAIN}.*zitibot_panda" 2>/dev/null || true
	[[ -n "${BASE_LOG}" && -f "${BASE_LOG}" ]] && rm -f "${BASE_LOG}"
	[[ -n "${GRIPPER_LOG}" && -f "${GRIPPER_LOG}" ]] && rm -f "${GRIPPER_LOG}"
}

on_interrupt() {
	echo "" >&2
	echo "Interrupted — shutting down." >&2
	cleanup
	trap - INT TERM EXIT
	exit 130
}

trap cleanup EXIT
trap on_interrupt INT TERM

base_alive() {
	[[ -n "${BASE_PID}" ]] && kill -0 "${BASE_PID}" 2>/dev/null
}

gripper_alive() {
	[[ -n "${GRIPPER_PID}" ]] && kill -0 "${GRIPPER_PID}" 2>/dev/null
}

wait_for_opensai_config() {
	local deadline=$((SECONDS + READY_TIMEOUT_SEC))
	echo "Waiting for OpenSai on Redis (up to ${READY_TIMEOUT_SEC}s)..."
	while (( SECONDS < deadline )); do
		if [[ -n "${LAUNCH_PID}" ]] && ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
			echo "Error: scripts/launch.sh exited before OpenSai was ready." >&2
			exit 1
		fi
		if command -v redis-cli >/dev/null 2>&1; then
			local cfg
			cfg="$(redis-cli GET "${CONFIG_REDIS_KEY}" 2>/dev/null || true)"
			if [[ "${cfg}" == *zitibot_panda.xml* ]]; then
				echo "   OpenSai ready (${cfg})."
				return 0
			fi
		fi
		sleep 0.5
	done
	echo "Error: timed out waiting for ${CONFIG_REDIS_KEY}." >&2
	exit 1
}

wait_for_redis_driver() {
	local deadline=$((SECONDS + DRIVER_READY_TIMEOUT_SEC))
	echo "Waiting for base redis_driver (up to ${DRIVER_READY_TIMEOUT_SEC}s)..."
	sleep 0.5
	while (( SECONDS < deadline )); do
		if ! base_alive; then
			echo "Error: redis_driver exited during startup." >&2
			show_log_tail "${BASE_LOG}" "redis_driver"
			exit 1
		fi
		if command -v redis-cli >/dev/null 2>&1; then
			local pose
			pose="$(redis-cli GET hb1::current_pose 2>/dev/null || true)"
			if [[ "${pose}" == \[* ]]; then
				echo "   hb1::current_pose OK (${pose})"
				return 0
			fi
		fi
		sleep 0.25
	done
	if ! base_alive; then
		echo "Error: redis_driver exited before ready." >&2
		show_log_tail "${BASE_LOG}" "redis_driver"
		exit 1
	fi
	if command -v redis-cli >/dev/null 2>&1; then
		echo "Error: redis_driver running but hb1::current_pose never appeared." >&2
		show_log_tail "${BASE_LOG}" "redis_driver"
		exit 1
	fi
	echo "   redis_driver process OK (no redis-cli to verify keys)."
}

preflight_cleanup_base

echo "=== launch_zitibot_full ==="
echo "Controller: ${CONTROLLER_PATH}"
echo "1) Starting OpenSai: scripts/launch.sh ${CONFIG_XML}"
(
	cd "${REPO_ROOT}"
	exec bash scripts/launch.sh "${CONFIG_XML}"
) &
LAUNCH_PID=$!
echo "   launch.sh pid ${LAUNCH_PID}"

wait_for_opensai_config

if [[ "${LAUNCH_GRIPPER}" == true ]]; then
	echo "2) Starting Franka gripper Redis driver via launch_gripper.sh..."
	GRIPPER_LOG="$(mktemp /tmp/franka_gripper.XXXXXX.log)"
	(
		cd "${GRIPPER_DIR}"
		exec bash launch_gripper.sh
	) >>"${GRIPPER_LOG}" 2>&1 &
	GRIPPER_PID=$!
	echo "   pid ${GRIPPER_PID}  log ${GRIPPER_LOG}"
	sleep 1
	if ! gripper_alive; then
		echo "Error: gripper driver failed to start." >&2
		show_log_tail "${GRIPPER_LOG}" "gripper driver"
		exit 1
	fi
else
	echo "2) Skipping gripper driver (--no-gripper)."
fi

if [[ "${LAUNCH_BASE}" == true ]]; then
	echo "3) Starting TidyBot base redis_driver (max_vel_xy=${MAX_VEL_XY} m/s)..."
	BASE_LOG="$(mktemp /tmp/redis_driver.XXXXXX.log)"
	(
		cd "${TIDYBOT_BASE}"
		exec "${PYTHON}" redis_driver.py \
			--max-vel-xy "${MAX_VEL_XY}" \
			--max-vel-yaw "${MAX_VEL_YAW}" \
			--max-accel-xy "${MAX_ACCEL_XY}" \
			--max-accel-yaw "${MAX_ACCEL_YAW}"
	) >>"${BASE_LOG}" 2>&1 &
	BASE_PID=$!
	echo "   pid ${BASE_PID}  log ${BASE_LOG}"
	if ! base_alive; then
		echo "Error: redis_driver failed to start." >&2
		show_log_tail "${BASE_LOG}" "redis_driver"
		exit 1
	fi
	sleep "${DRIVER_STARTUP_SEC}"
	wait_for_redis_driver
else
	echo "3) Skipping base driver (--no-base)."
fi

if [[ "${TUNE_MARKER_OFFSET}" == true ]]; then
	echo ""
	echo "Marker offset tune before controller:"
	echo "  This will command a short hb +X base move and print the measured offset."
	echo "  Ctrl+C aborts and shuts down all launched processes."
	echo ""
	(
		cd "${SCRIPT_DIR}/controllers"
		exec "${PYTHON}" tune_marker_offset.py
	)
fi

if [[ "${WAIT_FOR_SPACE}" == true ]]; then
	echo ""
	echo "Press SPACE in this terminal to start ${CONTROLLER_NAME}."
	echo "(Ctrl+C stops all launched processes.)"
	echo ""
	while IFS= read -r -n1 key; do
		if [[ "${key}" == " " ]]; then
			break
		fi
	done
else
	echo ""
	echo "Starting controller in:"
	for i in 3 2 1; do
		echo "  ${i}..."
		sleep 1
	done
	echo "  Go!"
	echo ""
fi

echo "4) Starting ${CONTROLLER_NAME}: ${PYTHON} ${CONTROLLER_PATH} ${CONTROLLER_ARGS[*]-}"
echo "   (Ctrl+C to stop controller and all drivers.)"

CTRL_EXIT=0
(
	cd "${CONTROLLER_DIR}"
	exec "${PYTHON}" "${CONTROLLER_SCRIPT}" "${CONTROLLER_ARGS[@]}"
) || CTRL_EXIT=$?

if [[ ${CTRL_EXIT} -ne 0 ]]; then
	echo "Controller exited with code ${CTRL_EXIT}." >&2
else
	echo "Controller exited." >&2
fi
echo "Shutting down." >&2
exit "${CTRL_EXIT}"
