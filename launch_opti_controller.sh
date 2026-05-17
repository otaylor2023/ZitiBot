#!/usr/bin/env bash
# Start tidybot_base/redis_driver.py, then opti_controller.py (mocap → hb1 goals).
#
# Only redis_driver.py is launched as a process. It imports Vehicle from
# base_controller.py — you do NOT run base_controller.py separately.
#
# Does NOT start OptiTrack / NatNet — run that manually (tidybot01::pos/ori on Redis).
#
# Prerequisites:
#   - Redis on localhost:6379
#   - Robot mini-PC with Phoenix / CAN (for redis_driver)
#
# Usage:
#   ./launch_opti_controller.sh
#   ./launch_opti_controller.sh --monitor
#   ./launch_opti_controller.sh --max-vel-xy 0.25
#   ./launch_opti_controller.sh   # default: straight line to Opti (-1.5, 1, 0.45) m, hold yaw
#   ./launch_opti_controller.sh -- --target-yaw-deg 90   # also command final heading
#   ./launch_opti_controller.sh -- --relative-goal   # 1.5 ft Motive +Y, then face −Y
#   ./launch_opti_controller.sh -- --rotate-only     # in-place rotate (use --relative-goal context)
#   ./launch_opti_controller.sh -- --relative-goal --goal-along lab-minus-x --goal-distance-ft 2
#   ./launch_opti_controller.sh -- --target-y 2.5     # override default Y only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIDYBOT_BASE="${SCRIPT_DIR}/controllers/tidybot_base"
OPTI_CONTROLLER="${SCRIPT_DIR}/controllers/opti_controller.py"
PYTHON="${PYTHON:-python3}"
DRIVER_STARTUP_SEC=3
DRIVER_READY_TIMEOUT_SEC=15

MAX_VEL_XY="${MAX_VEL_XY:-0.25}"
MAX_VEL_YAW="${MAX_VEL_YAW:-0.79}"
MAX_ACCEL_XY="${MAX_ACCEL_XY:-0.1}"
MAX_ACCEL_YAW="${MAX_ACCEL_YAW:-0.5}"

DRIVER_PID=""
OPTI_PID=""
OPTI_ARGS=()
DRIVER_LOG=""
BASE_PID_FILE="/tmp/tidybot2-base-controller.pid"

usage() {
	echo "Usage: $(basename "$0") [options] [opti_controller.py args...]"
	echo ""
	echo "  1) Starts controllers/tidybot_base/redis_driver.py (NOT base_controller.py)"
	echo "  2) Waits until redis_driver is alive and hb1::current_pose is on Redis"
	echo "  3) Starts opti_controller.py — only if step 2 succeeded"
	echo ""
	echo "Driver options:"
	echo "  --max-vel-xy M      planar max speed m/s (default: ${MAX_VEL_XY})"
	echo "  --max-vel-yaw R     max yaw rate rad/s (default: ${MAX_VEL_YAW})"
	echo "  --max-accel-xy A    planar accel m/s^2 (default: ${MAX_ACCEL_XY})"
	echo "  --max-accel-yaw A   yaw accel rad/s^2 (default: ${MAX_ACCEL_YAW})"
	exit 0
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		-h | --help) usage ;;
		--max-vel-xy) MAX_VEL_XY="$2"; shift 2 ;;
		--max-vel-yaw) MAX_VEL_YAW="$2"; shift 2 ;;
		--max-accel-xy) MAX_ACCEL_XY="$2"; shift 2 ;;
		--max-accel-yaw) MAX_ACCEL_YAW="$2"; shift 2 ;;
		--) shift; OPTI_ARGS+=("$@"); break ;;
		*) OPTI_ARGS+=("$1"); shift ;;
	esac
done

if [[ ! -f "${TIDYBOT_BASE}/redis_driver.py" ]]; then
	echo "Missing ${TIDYBOT_BASE}/redis_driver.py" >&2
	exit 1
fi
if [[ ! -f "${OPTI_CONTROLLER}" ]]; then
	echo "Missing ${OPTI_CONTROLLER}" >&2
	exit 1
fi

if command -v redis-cli >/dev/null 2>&1; then
	if ! redis-cli ping >/dev/null 2>&1; then
		echo "Error: redis-cli ping failed — start Redis first." >&2
		exit 1
	fi
else
	echo "Warning: redis-cli not found; cannot verify hb1::current_pose on Redis." >&2
fi

driver_alive() {
	[[ -n "${DRIVER_PID}" ]] && kill -0 "${DRIVER_PID}" 2>/dev/null
}

show_driver_log() {
	if [[ -n "${DRIVER_LOG}" && -f "${DRIVER_LOG}" ]]; then
		echo "--- redis_driver log (${DRIVER_LOG}) ---" >&2
		tail -n 80 "${DRIVER_LOG}" >&2 || true
		echo "--- end log ---" >&2
	fi
}

fail_driver() {
	local msg="$1"
	echo "Error: ${msg}" >&2
	show_driver_log
	cleanup
	exit 1
}

# Remove stale lock left when redis_driver was SIGKILL'd (atexit never ran).
remove_base_pid_file() {
	rm -f "${BASE_PID_FILE}"
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

# Kill orphaned drivers and clear zombie/stale PID locks before a new launch.
preflight_cleanup() {
	local old_pid=""
	if [[ -f "${BASE_PID_FILE}" ]]; then
		old_pid="$(tr -d '[:space:]' <"${BASE_PID_FILE}" 2>/dev/null || true)"
	fi
	if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
		if [[ -r "/proc/${old_pid}/status" ]] && grep -q '^State:[[:space:]]*Z' "/proc/${old_pid}/status"; then
			echo "Removing stale base-controller lock (zombie pid ${old_pid})"
			remove_base_pid_file
		else
			echo "Stopping previous redis_driver/base lock holder (pid ${old_pid})..."
			stop_process "${old_pid}" "previous base controller"
			remove_base_pid_file
		fi
	fi
	pkill -TERM -f "${TIDYBOT_BASE}/redis_driver.py" 2>/dev/null || true
	sleep 0.3
	pkill -KILL -f "${TIDYBOT_BASE}/redis_driver.py" 2>/dev/null || true
	remove_base_pid_file
}

cleanup() {
	stop_process "${OPTI_PID}" "opti_controller"
	stop_process "${DRIVER_PID}" "redis_driver"
	remove_base_pid_file
	if [[ -n "${DRIVER_LOG}" && -f "${DRIVER_LOG}" ]]; then
		rm -f "${DRIVER_LOG}"
	fi
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

wait_for_redis_driver() {
	local deadline=$((SECONDS + DRIVER_READY_TIMEOUT_SEC))
	echo "2) Waiting for redis_driver (up to ${DRIVER_READY_TIMEOUT_SEC}s)..."
	sleep 0.5
	while (( SECONDS < deadline )); do
		if ! driver_alive; then
			fail_driver "redis_driver exited during startup"
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
	if ! driver_alive; then
		fail_driver "redis_driver exited before ready"
	fi
	if command -v redis-cli >/dev/null 2>&1; then
		fail_driver "redis_driver running but hb1::current_pose never appeared on Redis"
	fi
	echo "   redis_driver process OK (no redis-cli to verify keys)."
	return 0
}

preflight_cleanup

echo "=== launch_opti_controller ==="
echo "1) Starting redis_driver (imports base_controller.Vehicle; max_vel_xy=${MAX_VEL_XY} m/s)..."

DRIVER_LOG="$(mktemp /tmp/redis_driver.XXXXXX.log)"
(
	cd "${TIDYBOT_BASE}"
	exec "${PYTHON}" redis_driver.py \
		--max-vel-xy "${MAX_VEL_XY}" \
		--max-vel-yaw "${MAX_VEL_YAW}" \
		--max-accel-xy "${MAX_ACCEL_XY}" \
		--max-accel-yaw "${MAX_ACCEL_YAW}"
) >>"${DRIVER_LOG}" 2>&1 &
DRIVER_PID=$!
echo "   pid ${DRIVER_PID}  log ${DRIVER_LOG}"

if ! driver_alive; then
	fail_driver "redis_driver failed to start"
fi

sleep "${DRIVER_STARTUP_SEC}"
wait_for_redis_driver

echo "3) Starting opti_controller..."
echo "   ${PYTHON} ${OPTI_CONTROLLER} ${OPTI_ARGS[*]-}"
echo ""
echo "Starting controller in:"
for i in 3 2 1; do
	echo "  ${i}..."
	sleep 1
done
echo "  Go!"
echo ""

"${PYTHON}" "${OPTI_CONTROLLER}" "${OPTI_ARGS[@]}"
OPTI_EXIT=$?

cleanup
exit "${OPTI_EXIT}"
