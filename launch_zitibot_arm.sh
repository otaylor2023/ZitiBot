#!/usr/bin/env bash
# Start OpenSai (zitibot_panda.xml) via scripts/launch.sh, the Franka gripper Redis
# driver, then a ZitiBot Python controller.
#
# Pass the controller as a file path (relative to ZitiBot/, relative to cwd, or absolute).
#
# Usage:
#   ./launch_zitibot_arm.sh controllers/touch_controller.py
#   ./launch_zitibot_arm.sh controllers/vision_controller.py
#   ./launch_zitibot_arm.sh controllers/vision_controller_new.py
#   ./launch_zitibot_arm.sh controllers/vision_controller.py -- --object mug
#   ./launch_zitibot_arm.sh controllers/vision_controller_new.py -- --no-goal-offset
#   ./launch_zitibot_arm.sh controllers/vision_controller_new.py -- --goal-offset-x 0.053 --goal-offset-z -0.10
#   ./launch_zitibot_arm.sh --wait controllers/touch_controller.py
#   ./launch_zitibot_arm.sh --no-gripper controllers/touch_controller.py
#
# Prerequisites:
#   - OpenSai built (bin/OpenSai_main), Redis reachable
#   - Gripper binary built unless --no-gripper (launched via launch_gripper.sh):
#       drivers/FrankaPanda/redis_driver/build/sai_franka_gripper_redis_driver
#       drivers/FrankaPanda/redis_driver/launch_gripper.sh
#   - launch_gripper.sh runs `sudo -S cpufreq-set ...` with the local password
#     baked in; if you change the user's password, update launch_gripper.sh too.
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

WAIT_FOR_SPACE=false
LAUNCH_GRIPPER=true
CONTROLLER_ARG=""
CONTROLLER_ARGS=()

GRIPPER_PID=""
GRIPPER_LOG=""

usage() {
	echo "Usage: $(basename "$0") [options] [--wait] <controller.py> [-- controller.py args...]"
	echo ""
	echo "Starts (in order):"
	echo "  1) scripts/launch.sh ${CONFIG_XML}  (OpenSai + Redis + web UI)"
	echo "  2) sai_franka_gripper_redis_driver  (unless --no-gripper)"
	echo "  3) Python controller (foreground, stdin for ENTER)"
	echo ""
	echo "Controller path resolution (first match wins):"
	echo "  1. exact path as given (absolute or relative to cwd)"
	echo "  2. relative to ZitiBot/ (so 'controllers/foo.py' works)"
	echo "  3. relative to ZitiBot/controllers/ (so 'foo.py' works)"
	echo ""
	echo "Options:"
	echo "  --wait          press SPACE in this terminal before starting the controller"
	echo "  --no-gripper    skip Franka gripper Redis driver"
	echo "  -h, --help"
	exit 0
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		-h | --help) usage ;;
		--wait) WAIT_FOR_SPACE=true; shift ;;
		--no-gripper) LAUNCH_GRIPPER=false; shift ;;
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
	echo "Error: controller file path required (e.g. controllers/vision_controller.py)." >&2
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
	echo "Missing ${OPENSAI_MAIN} — build OpenSai first (see repo README)." >&2
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

if command -v redis-cli >/dev/null 2>&1; then
	if ! redis-cli ping >/dev/null 2>&1; then
		echo "Warning: redis-cli ping failed — launch.sh may start redis-server." >&2
	fi
fi

LAUNCH_PID=""
CTRL_PID=""
CONFIG_REDIS_KEY="::sai-interfaces-webui::config_file_name"
READY_TIMEOUT_SEC=90

stop_tree() {
	local pid="$1"
	[[ -z "${pid}" ]] && return 0
	if ! kill -0 "${pid}" 2>/dev/null; then
		return 0
	fi
	# Stop children first (OpenSai_main, redis-server spawned by launch.sh, etc.).
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

gripper_alive() {
	[[ -n "${GRIPPER_PID}" ]] && kill -0 "${GRIPPER_PID}" 2>/dev/null
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
	if [[ -n "${CTRL_PID}" ]]; then
		stop_tree "${CTRL_PID}"
		CTRL_PID=""
	fi
	if [[ -n "${GRIPPER_PID}" ]]; then
		echo "Stopping gripper driver (pid ${GRIPPER_PID})..."
		stop_tree "${GRIPPER_PID}"
		GRIPPER_PID=""
	fi
	pkill -TERM -f "${GRIPPER_BIN}" 2>/dev/null || true
	if [[ -n "${LAUNCH_PID}" ]]; then
		stop_tree "${LAUNCH_PID}"
		LAUNCH_PID=""
	fi
	tmux kill-session -t interfaces_server 2>/dev/null || true
	pkill -TERM -f "${OPENSAI_MAIN}.*zitibot_panda" 2>/dev/null || true
	[[ -n "${GRIPPER_LOG}" && -f "${GRIPPER_LOG}" ]] && rm -f "${GRIPPER_LOG}"
}

on_interrupt() {
	echo "" >&2
	echo "Interrupted — stopping controller and OpenSai." >&2
	cleanup
	trap - INT TERM EXIT
	exit 130
}

trap cleanup EXIT
trap on_interrupt INT TERM

wait_for_opensai_config() {
	local deadline=$((SECONDS + READY_TIMEOUT_SEC))
	echo "Waiting for OpenSai config on Redis (up to ${READY_TIMEOUT_SEC}s)..."
	while (( SECONDS < deadline )); do
		if [[ -n "${LAUNCH_PID}" ]] && ! kill -0 "${LAUNCH_PID}" 2>/dev/null; then
			echo "Error: scripts/launch.sh exited before OpenSai was ready." >&2
			exit 1
		fi
		if command -v redis-cli >/dev/null 2>&1; then
			local cfg
			cfg="$(redis-cli GET "${CONFIG_REDIS_KEY}" 2>/dev/null || true)"
			if [[ "${cfg}" == *zitibot_panda.xml* ]]; then
				echo "OpenSai ready (${cfg})."
				return 0
			fi
		fi
		sleep 0.5
	done
	echo "Error: timed out waiting for ${CONFIG_REDIS_KEY} to contain zitibot_panda.xml." >&2
	exit 1
}

echo "=== launch_zitibot_arm ==="
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

if [[ "${WAIT_FOR_SPACE}" == true ]]; then
	echo ""
	echo "Press SPACE in this terminal to start ${CONTROLLER_NAME} (${CONTROLLER_SCRIPT})."
	echo "(Ctrl+C stops the controller, gripper driver, and OpenSai.)"
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

echo "3) Starting ${CONTROLLER_NAME}: ${PYTHON} ${CONTROLLER_PATH} ${CONTROLLER_ARGS[*]-}"
echo "   (Ctrl+C to stop the controller, gripper driver, and OpenSai.)"

# Run the controller in the FOREGROUND so it inherits this terminal's stdin.
# Backgrounding it (with &) makes Python's readline() see EOF immediately on
# many systems, which causes interactive controllers to misbehave.
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
echo "Stopping OpenSai." >&2
