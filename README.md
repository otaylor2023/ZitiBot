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
