# Shape Based Matching Sample

This repository is a clean rebuild around `https://github.com/meiqua/shape_based_matching`.

It uses:

- OpenCV for image loading and visualization
- upstream `line2Dup` from `shape_based_matching`
- a two-stage workflow:
  - `train`
  - `match`

The sample assets are:

- `assets/template.bmp`
- `assets/scene.bmp`

## Prerequisites

- CMake 3.25 or newer
- A C++20 compiler
- OpenCV installed and discoverable by CMake

If CMake cannot find OpenCV automatically, configure with `-DOpenCV_DIR=...`.

## Configure and Build

```bash
cmake -S . -B build -DOpenCV_DIR=D:/proj/3rdParty/opencv410/build
cmake --build build --config Debug
```

## Train

```bash
build/Debug/shape_match_sample train assets/template.bmp models/default
```

Train with an explicit training ROI:

```bash
build/Debug/shape_match_sample train assets/template.bmp models/default --train-roi x,y,w,h
```

Training writes:

- `models/default/template.yaml`
- `models/default/info.yaml`
- `models/default/model_meta.yaml`

## Match

```bash
build/Debug/shape_match_sample match assets/scene.bmp models/default output/match_overlay.png
```

Match with an explicit search ROI:

```bash
build/Debug/shape_match_sample match assets/scene.bmp models/default output/match_overlay.png --search-roi x,y,w,h
```

The overlay image is written to the requested path. If no overlay path is provided, the default is `output/match_overlay.png`.

## ROI Rules

- `--train-roi` and `--search-roi` use `x,y,w,h`
- both are optional
- if omitted, the whole image is used
- invalid ROI values fail immediately

## Test

```bash
ctest --test-dir build -C Debug --output-on-failure
```
