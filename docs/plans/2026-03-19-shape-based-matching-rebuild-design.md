# Shape Based Matching Rebuild Design

The old `OpenFDCM` project was discarded completely. This repository is rebuilt from scratch around `shape_based_matching`, with `assets/template.bmp` and `assets/scene.bmp` preserved from the previous workspace.

The new project exposes two commands:

- `train <template_image> <model_dir> [class_id] [--train-roi x,y,w,h]`
- `match <scene_image> <model_dir> [class_id] [overlay_output] [--search-roi x,y,w,h]`

Training writes:

- `template.yaml`
- `info.yaml`
- `model_meta.yaml`

Matching loads those files, runs `line2Dup::Detector::match(...)`, maps ROI-local hits back to full-image coordinates, and writes an overlay image.

The implementation uses OpenCV for image I/O and visualization, and compiles the upstream `line2Dup.cpp` directly through the local `CMakeLists.txt` instead of relying on the upstream build script.
