#!/usr/bin/env bash
# Start OpenSai (zitibot_panda.xml) via scripts/launch.sh, then a ZitiBot Python controller.
#
# Controllers live under ZitiBot/controllers/ (base / opti not included here).
#
# Usage:
#   ./launch_zitibot_panda.sh touch
#   ./launch_zitibot_panda.sh vision
#   ./launch_zitibot_panda.sh vision -- --object mug
#   ./launch_zitibot_panda.sh --wait touch
#
# Prerequisites: OpenSai built (bin/OpenSai_main), Redis reachable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTROLLERS_DIR="${SCRIPT_DIR}/controllers"
CONFIG_XML="config_folder/xml_config_files/zitibot_panda.xml"
LAUNCH_SH="${REPO_ROOT}/scripts/launch.sh"
OPENSAI_MAIN="${REPO_ROOT}/bin/OpenSai_main"
PYTHON="${PYTHON:-python3}"

WAIT_FOR_SPACE=false
CONTROLLER_NAME=""
CONTROLLER_ARGS=()

declare -A CONTROLLER_SCRIPTS=(
	[touch]="touch_controller.py"
	[vision]="vision_controller_new.py"
)

usage() {
	echo "Usage: $(basename "$0") [--wait] <controller> [-- controller.py args...]"
	echo ""
	echo "Starts: scripts/launch.sh ${CONFIG_XML}"
	echo "Then runs a Python controller from ZitiBot/controllers/."
	echo ""
	echo "Controllers:"
	for name in "${!CONTROLLER_SCRIPTS[@]}"; do
		echo "  ${name}  ->  ${CONTROLLER_SCRIPTS[${name}]}"
	done | sort
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
			if [[ -z "${CONTROLLER_NAME}" ]]; then
				CONTROLLER_NAME="$1"
				shift
			else
				CONTROLLER_ARGS+=("$1")
				shift
			fi
			;;
	esac
done

if [[ -z "${CONTROLLER_NAME}" ]]; then
	echo "Error: controller name required (e.g. touch, vision)." >&2
	echo "Run: $(basename "$0") --help" >&2
	exit 1
fi

CONTROLLER_SCRIPT="${CONTROLLER_SCRIPTS[${CONTROLLER_NAME}]-}"
if [[ -z "${CONTROLLER_SCRIPT}" ]]; then
	echo "Error: unknown controller '${CONTROLLER_NAME}'." >&2
	echo "Available: ${!CONTROLLER_SCRIPTS[*]}" >&2
	exit 1
fi

CONTROLLER_PATH="${CONTROLLERS_DIR}/${CONTROLLER_SCRIPT}"
if [[ ! -f "${CONTROLLER_PATH}" ]]; then
	echo "Missing ${CONTROLLER_PATH}" >&2
	exit 1
fi
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
(
	cd "${CONTROLLERS_DIR}"
	exec "${PYTHON}" "${CONTROLLER_SCRIPT}" "${CONTROLLER_ARGS[@]}"
) &
CTRL_PID=$!

# Exit when either launch.sh or the controller stops.
while kill -0 "${LAUNCH_PID}" 2>/dev/null && kill -0 "${CTRL_PID}" 2>/dev/null; do
	if IFS= read -r -n1 -t 1 key; then
		if [[ "${key}" == "q" || "${key}" == "Q" ]]; then
			echo "Quit key pressed; shutting down." >&2
			break
		fi
	fi
done

if ! kill -0 "${LAUNCH_PID}" 2>/dev/null && kill -0 "${CTRL_PID}" 2>/dev/null; then
	echo "OpenSai exited; stopping controller." >&2
fi
if ! kill -0 "${CTRL_PID}" 2>/dev/null && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
	echo "Controller exited; stopping OpenSai." >&2
fi
