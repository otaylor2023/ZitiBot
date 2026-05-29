# ZitiBot: Robotic Chef for Baked Ziti

Final project for **CS225A**: a **mobile manipulator** (differential-drive base and **Franka Panda** arm) programmed to cook **baked ziti!** The same controller and visualizer are used in **simulation** and can be deployed on the **physical robot** when wired to the Redis interface and hardware drivers.

This repository is a **standalone** CMake project. It depends on the same **OpenSai / SAI** libraries (CHAI3D, Redis, SaiModel, SaiSimulation, SaiGraphics, SaiPrimitives, etc.) as the course stack; install those per course instructions before building.

## URDF assets

Robot and world files live under **`urdf_models/`** at the **ZitiBot** repo root (`mmp_panda/`, `panda/`, `test_objects/`). That is the supported layout for this standalone project (no dependency on `cs225a/urdf_models` at runtime). CMake defaults to this folder; to use another tree, configure with:

```bash
cmake -S . -B build -DURDF_MODELS_FOLDER=/absolute/path/to/urdf_models
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4) \
  --target controller_zitibot_mmp_panda simviz_zitibot_mmp_panda
```

Or use the helper script from the repo root:

```bash
./build_zitibot.sh              # configure + build the two executables
./build_zitibot.sh --all        # build every target in this project
```

Optional environment: **`JOBS`**, **`CMAKE_BUILD_TYPE`**.

Binaries: **`bin/zitibot_mmp_example/simviz_zitibot_mmp_panda`** and **`controller_zitibot_mmp_panda`**.

**Redis** must be running before starting the sim or controller.

## Launch

```bash
./launch_zitibot.sh           # visualizer + controller (no build step)
./launch_zitibot.sh --wait    # wait for SPACE before starting the controller
./launch_zitibot.sh --build   # configure + build, then launch
```

- **Ctrl+C** in the terminal stops the visualizer and controller.
- **Q** with keyboard focus in the **graphics window** closes the visualizer; the script stops the controller and exits.

`./launch_zitibot.sh --help` lists options.

## Vision

Python utilities for the onboard Intel RealSense camera live under **`python_control/vision/`** (library code used by scripts in **`python_control/`**).

Install Python deps (works in a venv or conda env):

```bash
pip install -r requirements.txt
```

Note: `pyrealsense2` ships prebuilt wheels for Linux x86_64 and Windows. On macOS / Linux ARM you'll need to build `librealsense` from source.

### Stream the camera

```bash
python python_control/vision/test_camera.py                # color + depth side by side
python python_control/vision/test_camera.py --no-depth     # color only
python python_control/vision/test_camera.py --fps 15       # drop fps if USB bandwidth is tight
```

`q` or `Esc` to quit.

### Grasp demo (parallel-jaw / 2-finger gripper)

`python_control/vision/grasp_demo.py` streams the RealSense and, on SPACE, runs a heatmap grasp predictor on the current RGB-D frame. Two pretrained models are bundled, selected with `--model`:

| `--model` | Network | Input | Default size | Source |
|-----------|---------|-------|--------------|--------|
| `ggcnn2` (default) | GG-CNN2 (Morrison et al., RSS 2018) | depth only | 300×300 | [dougsm/ggcnn](https://github.com/dougsm/ggcnn) |
| `grconvnet` | GR-ConvNet v3 (Kumra et al., IROS 2020) | RGB-D | 224×224 | [skumra/robotic-grasping](https://github.com/skumra/robotic-grasping) |

Both are single-pass heatmap networks and run real-time on CPU. GR-ConvNet is heavier and a bit stronger on Cornell; GG-CNN2 is tiny.

One-time: fetch whichever model's pretrained Cornell weights you want.

```bash
bash python_control/vision/ggcnn/weights/download_weights.sh        # ~1 MB
bash python_control/vision/grconvnet/weights/download_weights.sh    # ~30 MB
```

Run:

```bash
python python_control/vision/grasp_demo.py                          # ggcnn2 (default)
python python_control/vision/grasp_demo.py --model grconvnet
python python_control/vision/grasp_demo.py --model grconvnet --crop 400 --fps 15
```

Keys:

- **SPACE** -- run the selected model on the current frame; pop a window with a jet-colormapped quality heatmap and the top antipodal grasp (red = gripper plates, green = opening width).
- **s** -- save the latched grasp overlay as `grasp_<model>_<timestamp>.png` in the cwd.
- **q** / **Esc** -- quit.

### Gemini ER pointing + robot (OpenSai Redis)

**`python_control/vision_controller.py`** runs the live RealSense + **Gemini Robotics-ER** overlay, then on **ENTER** writes OpenSai Franka **cartesian_task** goal position/orientation to Redis (same JSON keys as `python_control/touch_controller.py`). **SPACE** queries Gemini; **s** saves the overlay; **q** / **Esc** quits.

For a **single saved image** (2D keypoints only, no camera, no Redis), use:

```bash
python python_control/gemini_image_cli.py --image path/to/scene.jpg
```

Gemini/geometry helpers live under **`python_control/vision/gemini_pointing.py`** (library only — no Redis).

Install (covers `google-genai` and the optional `python-dotenv` auto-loader):

```bash
pip install -r requirements.txt
```

Get an API key from [Google AI Studio](https://aistudio.google.com/) and put it in a `.env` file at the repo root — the library auto-loads it on every run (`.env` is gitignored):

```bash
echo 'GEMINI_API_KEY=your-key-here' > .env
```

A real `GEMINI_API_KEY` / `GOOGLE_API_KEY` environment variable also works if you prefer.

Live camera + robot goals (from the OpenSai / ZitiBot repo root):

```bash
python python_control/vision_controller.py
python python_control/vision_controller.py --object "pasta pot"
python python_control/vision_controller.py --prompt "Point to the rim of the bowl."
```

Keys (live): **SPACE** = query Gemini, **ENTER** = send cartesian goal, **s** = save overlay, **q** / **Esc** = quit.

### Grasp and pour (Gemini + Franka gripper)

**`ZitiBot/controllers/grasp_and_pour_controller.py`** is the interactive real-robot flow (no mobile base): same Gemini + RealSense UI as `vision_controller.py`, but **ENTER** is two-step — first descend to the latched pick with gripper **open**, then **close**, **lift**, and **pour** (90° about world +Y, ported from `zitibot_mmp_panda/controller.cpp`).

Prerequisites:

- Redis, OpenSai web UI on **`zitibot_panda.xml`**, **`cartesian_controller`** active
- Franka arm Redis driver and Franka **gripper** driver (`opensai::FrankaRobot::gripper::desired_width`, etc.)
- RealSense + `GEMINI_API_KEY` (see above)

```bash
python ZitiBot/controllers/grasp_and_pour_controller.py
python ZitiBot/controllers/grasp_and_pour_controller.py --lift-dz 0.12 --object bowl
```

Keys: **SPACE** = Gemini + latch pick | **ENTER** = grasp (open) then lift+pour | **s** = save overlay | **q** / **Esc** = quit.

The sim auto-FSM remains **`launch_zitibot.sh`** / `controller_zitibot_mmp_panda` (C++); use this script on hardware with OpenSai.
