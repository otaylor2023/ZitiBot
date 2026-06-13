# EggBot: A Robot That Cooks Scrambled Eggs

Final project for **CS 225A** (Stanford). EggBot is a **mobile manipulator** — a
**Franka Emika Panda / FR3** arm with a **Franka Hand** gripper riding on a **TidyBot**
holonomic base — that cooks scrambled eggs end-to-end: it grabs an egg, cracks it,
clears the shell, repeats for a second egg, whisks, pours into a pan, moves the pan
to the stove, picks up a ladle, and scrambles.

The project started in the **OpenSai** simulator and moved to the real robot. Grasps
are chosen on-line with **Gemini Robotics-ER 1.6**; motion runs through OpenSai's
task-space Cartesian controller over Redis.

Project website: see **`docs/`** or the published
[GitHub Pages site](https://otaylor2023.github.io/EggBot/).

---

## Where this lives inside OpenSai

This repo is meant to be checked out **as a subfolder of an OpenSai tree**:

```
OpenSai/
├── bin/OpenSai_main                       # OpenSai sim/controller binary
├── config_folder/xml_config_files/
│   └── zitibot_panda.xml                   # robot + cartesian_controller config
├── drivers/FrankaPanda/redis_driver/       # Franka arm + gripper Redis drivers
├── scripts/launch.sh                       # OpenSai launcher
└── ZitiBot/                                # ← this repo
    ├── controllers/                        # Python real-robot controllers (Redis)
    ├── zitibot_mmp_panda/                  # standalone C++ sim (sim-only FSM)
    ├── urdf_models/                        # robot + world URDFs
    ├── docs/                               # project website (GitHub Pages)
    └── launch_zitibot_*.sh                 # orchestration scripts
```

The launch scripts resolve `REPO_ROOT` as the **parent** of `ZitiBot/` (i.e. the OpenSai
root) and expect `bin/OpenSai_main`, the `zitibot_panda.xml` config, and the Franka
drivers to exist there. The Python controllers talk to whatever OpenSai publishes on
Redis, so they are agnostic to sim vs. real as long as the keys are populated.

All OpenSai/SAI dependencies (CHAI3D, Redis, SaiModel, SaiSimulation, SaiGraphics,
SaiPrimitives, Eigen) come from the course stack — install those per course instructions
before building anything here.

---

## Quick start (real robot)

```bash
# 1. Start OpenSai (arm + cartesian_controller), the Franka gripper driver,
#    and the TidyBot base driver, then a Python controller:
./launch_zitibot_full.sh controllers/grasp_and_pour_controller.py

# arm only (no mobile base), e.g. for bench testing:
./launch_zitibot_arm.sh controllers/egg_crack_controller.py
```

- `--wait` holds the controller until you press SPACE.
- `--no-base` / `--no-gripper` skip those drivers.
- Anything after `--` is forwarded to the Python controller:
  `./launch_zitibot_full.sh controllers/pour_and_move_controller.py -- --grasp-opti-x -2.52`

Prerequisites the scripts assume: `bin/OpenSai_main` built, the Franka gripper driver
built under `drivers/FrankaPanda/redis_driver/`, OptiTrack/Motive publishing
`tidybot01::*` to Redis (run separately), and **Redis running**.

---

## Control architecture

Motion is a **task-space Cartesian controller** owned by OpenSai. Python controllers
write goals to Redis under the `cartesian_controller::cartesian_task` namespace:

```
opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_position
opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::goal_orientation
opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_position
opensai::controllers::FrankaRobot::cartesian_controller::cartesian_task::current_orientation
```

Design notes:

- **Per-task gains/speeds.** Precise grasps (egg cracker, tongs; ~0.5 cm tolerance) run
  with higher gains and lower speeds; carrying/pouring loosens gains and moves faster.
- **Joint space for shaking.** The vigorous shake/scramble motions are commanded directly
  in joint space — cleaner, more repeatable oscillations than chasing a Cartesian goal.
- **Gated state machines.** The interactive real-robot controllers (e.g.
  `egg_crack_controller.py`) advance one phase per **ENTER** press
  (`ABOVE_PICK → AT_PICK → GRASPED → LIFTED → SQUEEZED → ABOVE_DROP → AT_DROP → DONE`),
  which made debugging on hardware safe.

### Python controllers (`controllers/`)

| File | Role |
|------|------|
| `vision_controller_new.py` | Live RealSense + Gemini overlay; SPACE = query Gemini & latch a grasp, ENTER = publish the Cartesian goal. Current vision entry point. |
| `grasp_and_pour_controller.py` | Interactive grasp → close → lift → pour (90° about world +Y). |
| `grasp_pour_vision_base_controller.py` | Grasp-and-pour with the TidyBot base in the loop. |
| `egg_crack_controller.py` | Arm + gripper egg-cracker sequence with the force schedule below. |
| `pour_and_move_controller.py`, `mixing_*`, `stirring_controller.py` | Per-primitive motions reused across the pipeline. |
| `gemini_image_cli.py` | Run Gemini on a single saved image (2D keypoints only — no camera, no Redis). |
| `tidybot_base/` | TidyBot holonomic base: Redis driver, SE(2) planning, OptiTrack mocap nav. |

---

## Vision-guided grasping

We use **Gemini Robotics-ER 1.6** (`gemini-robotics-er-1.6-preview`) to pick grasp points
from the wrist-mounted **RealSense D405** RGB-D stream. The geometry/library code (no
Redis, no robot) lives in **`controllers/vision/gemini_pointing.py`**:

1. Send the RGB frame + a prompt to Gemini; parse the JSON keypoint(s) it returns
   (`parse_points`).
2. Sample depth at each pixel as a **median over a small patch** so a few bad depth
   readings don't throw it off (`sample_depth_median`).
3. Back-project pixel + depth into the camera frame with the RealSense intrinsics
   (`deproject_pixel_to_cam` / `lift_points_to_3d`).

The controller then chains the transform to world frame:

```
p_base = T_base_flange · T_flange_camera · p_camera
```

Two grasp strategies:

- **Single-point grasps (egg, tools).** Gemini marks one point; the gripper stays
  tool-down and closes at that 3D position. Used for picking up the egg with tongs.
- **Two-point rim grasps (bowl).** Gemini returns *two* rim points. Both are lifted to
  3D; the first is the grasp position, and the vector between them is projected onto the
  table plane. Its yaw (`rim_yaw_rotation_rad` → `arctan2`) is applied as a world-Z
  rotation so the jaws line up across the rim before pinching and pouring.

### Camera utilities

```bash
python controllers/vision/test_camera.py            # color + depth side by side
python controllers/vision/grasp_demo.py             # GG-CNN2 / GR-ConvNet heatmap grasps (legacy)
```

`grasp_demo.py` bundles two pretrained heatmap grasp nets (GG-CNN2 depth-only and
GR-ConvNet v3 RGB-D) selectable with `--model`; download their Cornell weights via the
`weights/download_weights.sh` scripts under `controllers/vision/ggcnn|grconvnet`.

---

## Force control

The Franka Hand's grasping force runs from about **30 N to 140 N**, but a raw egg fractures
near **20 N** — below what the gripper can reliably command. So:

- **Egg:** picked up with **tongs**, whose compliant arms spread the contact force enough
  to lift the egg intact.
- **Egg cracker:** grasped at low force, then ramped toward max over **three consecutive
  squeezes** to break the shell, then released so the shell halves fall into the bowl. See
  the `LIFTED → SQUEEZED` phase and the `lift_force_n` / `crack_force_n` /
  `crack_unlatch_m` parameters in `egg_crack_controller.py`.
- **Whisk:** held lightly but firmly, squeezed to switch its mechanism on, loosened to
  stop.

---

## Simulation (OpenSai)

The sim-only prototype is a **standalone CMake C++ project** under `zitibot_mmp_panda/`
(`controller.cpp`, `simviz.cpp`, `world_mmp_panda.urdf`). It links the same SAI libraries
as the course stack.

```bash
# configure + build the two executables
./build_zitibot.sh
# or directly:
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target controller_zitibot_mmp_panda simviz_zitibot_mmp_panda
```

URDFs default to `urdf_models/` at the repo root; override with
`-DURDF_MODELS_FOLDER=/abs/path`. Redis must be running before launching. Run the built
binaries (`simviz_zitibot_mmp_panda` for the visualizer, `controller_zitibot_mmp_panda`
for the sim FSM) from `bin/`. On hardware, use `launch_zitibot_full.sh` /
`launch_zitibot_arm.sh` with the Python controllers instead.

---

## Python setup

```bash
pip install -r requirements.txt
```

`pyrealsense2` ships prebuilt wheels for Linux x86_64 / Windows; on macOS or Linux ARM
build `librealsense` from source.

Gemini needs an API key from [Google AI Studio](https://aistudio.google.com/). Put it in a
`.env` at the repo root — it's auto-loaded on every run and is gitignored:

```bash
echo 'GEMINI_API_KEY=your-key-here' > .env
```

A `GEMINI_API_KEY` / `GOOGLE_API_KEY` environment variable works too.

---

## Project website (`docs/`)

The site is published with GitHub Pages from the **`docs/`** folder
(**Settings → Pages → Deploy from a branch → `main` / `/docs`**).

```bash
# local preview
python3 -m http.server 8080 --directory docs   # http://localhost:8080
```

Media in `docs/assets/` is transcoded (H.264, mostly audio-stripped) from raw clips in
`assets/edited_videos/` (gitignored). Rebuild after swapping clips:

```bash
bash scripts/build_site_media.sh               # requires ffmpeg
```

Grasp overlays are copied from `logs2/` by default; override with
`ZITIBOT_LOGS=/path/to/logs`.
