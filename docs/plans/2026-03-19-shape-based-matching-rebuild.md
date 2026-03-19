# Shape Based Matching Rebuild Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a fresh C++ sample that trains and matches templates with `shape_based_matching`, using OpenCV for image I/O and ROI visualization.

**Architecture:** A local wrapper library owns ROI handling, model persistence, and overlay drawing. The CLI calls into that library through `train` and `match` subcommands, while tests exercise both the library and the real CLI.

**Tech Stack:** CMake, C++20, OpenCV, shape_based_matching, CTest

---

### Task 1: Preserve assets and rebuild the repository

- Preserve `template.bmp` and `scene.bmp`
- Delete the previous workspace contents, including `.git`
- Recreate a new Git repository
- Restore the bitmap assets into `assets/`

### Task 2: Build training support

- Add `FetchContent` for `shape_based_matching`
- Compile upstream `line2Dup.cpp`
- Implement `train_model(...)`
- Persist `template.yaml`, `info.yaml`, and `model_meta.yaml`
- Add smoke tests for training and training ROI

### Task 3: Build matching support

- Implement `match_model(...)`
- Support search ROI remapping into full-image coordinates
- Write match overlays
- Add smoke tests for matching and search ROI

### Task 4: Add the CLI and documentation

- Implement `shape_match_sample train ...`
- Implement `shape_match_sample match ...`
- Add CLI tests
- Write README usage examples
