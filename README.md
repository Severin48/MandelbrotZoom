# MellowSim

GPU-accelerated Mandelbrot explorer written in modern C++ with **OpenCL** for compute and **OpenCV** for display & input. Navigate smoothly, zoom interactively, and export high-resolution frames.

---

## Demo

### Example Image
![Example output placeholder](images/ExampleImage.png)


---

## Features

- Interactive Mandelbrot zooming with mouse (GPU-accelerated with OpenCL)
- Smooth progressive refinement (increasing iteration limits)
- Live preview window via OpenCV
- Save full-resolution outputs to `./output/`
- Optional **guided zoom** playback from files in `./zooms/`


### Example Guided Zoom
![Example output placeholder](images/GuidedZoomExample.gif)

---

## Requirements

- **C++17 or newer** toolchain (GCC/Clang on Linux, MSVC on Windows)
- **CMake ≥ 3.16**
- **OpenCL runtime** (GPU vendor driver)
- **OpenCV ≥ 4.x** (core, imgproc, highgui)

### Recommended tooling
- Linux: `build-essential`, `cmake`, `ocl-icd-opencl-dev`, `opencl-headers`, `clinfo`, `libopencv-dev`
- Windows: Visual Studio Build Tools (MSVC), CMake, **vcpkg** for OpenCV & OpenCL-ICD-Loader

---

## Quick start

### Linux (Debian/Ubuntu)
```bash
sudo apt update
sudo apt install -y build-essential cmake git \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libopencv-dev

# verify OpenCL sees your device
clinfo | head -n 20

# build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# run
./build/bin/MellowSim
