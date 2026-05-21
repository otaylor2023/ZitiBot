#!/usr/bin/env bash
# Start OpenSai (zitibot_panda.xml) via scripts/launch.sh, then a ZitiBot Python controller.
#
# Pass the controller as a file path (relative to ZitiBot/, relative to cwd, or absolute).
#
# Usage:
#   ./launch_zitibot_panda.sh controllers/touch_controller.py
#   ./launch_zitibot_panda.sh controllers/vision_controller.py
#   ./launch_zitibot_panda.sh controllers/vision_controller_new.py
#   ./launch_zitibot_panda.sh controllers/vision_controller.py -- --object mug
#   ./launch_zitibot_panda.sh controllers/vision_controller_new.py -- --no-goal-offset
#   ./launch_zitibot_panda.sh controllers/vision_controller_new.py -- --goal-offset-x 0.053 --goal-offset-z -0.10
#   ./launch_zitibot_panda.sh --wait controllers/touch_controller.py
#
# Prerequisites: OpenSai built (bin/OpenSai_main), Redis reachable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_XML="config_folder/xml_config_files/zitibot_panda.xml"
LAUNCH_SH="${REPO_ROOT}/scripts/launch.sh"
OPENSAI_MAIN="${REPO_ROOT}/bin/OpenSai_main"
PYTHON="${PYTHON:-python3}"

WAIT_FOR_SPACE=false
CONTROLLER_ARG=""
CONTROLLER_ARGS=()

usage() {
	echo "Usage: $(basename "$0") [--wait] <controller.py> [-- controller.py args...]"
	echo ""
	echo "Starts: scripts/launch.sh ${CONFIG_XML}"
	echo "Then runs the given Python controller file."
	echo ""
	echo "Controller path resolution (first match wins):"
	echo "  1. exact path as given (absolute or relative to cwd)"
	echo "  2. relative to ZitiBot/ (so 'controllers/foo.py' works)"
	echo "  3. relative to ZitiBot/controllers/ (so 'foo.py' works)"
	echo ""
	echo "Options:"
	echo "  --wait    press SPACE in this terminal before starting the controller"
	echo "  -h, --help"
	exit 0
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		-h | --help) usage ;;
		--wait) WAIT_FOR_SPACE=true; shift ;;
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

cleanup() {
	if [[ -n "${CTRL_PID}" ]]; then
		stop_tree "${CTRL_PID}"
		CTRL_PID=""
	fi
	if [[ -n "${LAUNCH_PID}" ]]; then
		stop_tree "${LAUNCH_PID}"
		LAUNCH_PID=""
	fi
	tmux kill-session -t interfaces_server 2>/dev/null || true
	pkill -TERM -f "${OPENSAI_MAIN}.*zitibot_panda" 2>/dev/null || true
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

echo "=== launch_zitibot_panda ==="
echo "Controller: ${CONTROLLER_PATH}"
echo "1) Starting OpenSai: scripts/launch.sh ${CONFIG_XML}"
(
	cd "${REPO_ROOT}"
	exec bash scripts/launch.sh "${CONFIG_XML}"
) &
LAUNCH_PID=$!
echo "   launch.sh pid ${LAUNCH_PID}"

wait_for_opensai_config

if [[ "${WAIT_FOR_SPACE}" == true ]]; then
	echo ""
	echo "Press SPACE in this terminal to start ${CONTROLLER_NAME} (${CONTROLLER_SCRIPT})."
	echo "(Ctrl+C stops OpenSai and exits.)"
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

echo "2) Starting ${CONTROLLER_NAME}: ${PYTHON} ${CONTROLLER_PATH} ${CONTROLLER_ARGS[*]-}"
echo "   (Ctrl+C to stop the controller and OpenSai.)"

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
