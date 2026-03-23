# Shape Matcher DLL Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the current shape-based matching sample into a reusable C++17 DLL that exposes a `ShapeMatcher` class using `cv::Mat`, while preserving the CLI as a thin wrapper.

**Architecture:** The public library surface moves to an exported `ShapeMatcher` class plus shared data types in public headers. The DLL owns the detector and model metadata in memory, supports save/load to the existing YAML layout, and the CLI calls into that DLL instead of owning matching logic directly.

**Tech Stack:** CMake, C++17, OpenCV, shape_based_matching, CTest, MSVC DLL exports

---

### Task 1: Add DLL API regression tests first

**Files:**
- Create: `tests/dll_train_match.cpp`
- Create: `tests/dll_persistence.cpp`
- Modify: `CMakeLists.txt`

**Step 1: Write the failing test**

Add `tests/dll_train_match.cpp` that uses the planned public API:

```cpp
#include "shape_match_sample/shape_matcher.hpp"

#include <filesystem>

#include <opencv2/imgcodecs.hpp>

int main() {
    const auto source_dir = std::filesystem::path{SHAPE_MATCH_SAMPLE_SOURCE_DIR};
    const cv::Mat templ = cv::imread((source_dir / "assets" / "template.bmp").string(), cv::IMREAD_COLOR);
    const cv::Mat scene = cv::imread((source_dir / "assets" / "scene.bmp").string(), cv::IMREAD_COLOR);

    shape_match_sample::ShapeMatcher matcher;
    matcher.train(templ);
    const auto result = matcher.match(scene);
    return result.matches.empty() ? 1 : 0;
}
```

Add `tests/dll_persistence.cpp` that trains, saves, loads into a second matcher, and matches again.

**Step 2: Run test to verify it fails**

Run: `cmake --build build --config Debug --target dll_train_match`

Expected: compile failure because `shape_match_sample/shape_matcher.hpp` and the class do not exist yet.

**Step 3: Register the tests in CMake**

Modify `CMakeLists.txt` to add executables for `dll_train_match` and `dll_persistence`, link them to the future shared library target, and add `ctest` registrations.

**Step 4: Run test to verify it still fails for the right reason**

Run: `cmake -S . -B build -DOpenCV_DIR=D:/proj/3rdParty/opencv410/build`

Expected: configure succeeds, but the build still fails because the public DLL API is not implemented yet.

**Step 5: Commit**

```bash
git add CMakeLists.txt tests/dll_train_match.cpp tests/dll_persistence.cpp
git commit -m "test: add dll api regression coverage"
```

### Task 2: Introduce the exported public API and shared library target

**Files:**
- Create: `include/shape_match_sample/export.hpp`
- Create: `include/shape_match_sample/types.hpp`
- Create: `include/shape_match_sample/shape_matcher.hpp`
- Modify: `CMakeLists.txt`

**Step 1: Write the failing test**

Extend `tests/dll_train_match.cpp` so it also checks:

```cpp
if (matcher.empty()) {
    return 1;
}
```

This should still fail to compile until the class skeleton and export macro exist.

**Step 2: Run test to verify it fails**

Run: `cmake --build build --config Debug --target dll_train_match`

Expected: compile failure referencing missing `ShapeMatcher::empty()`.

**Step 3: Write minimal implementation**

Add:

- `export.hpp` with `SHAPE_MATCH_SAMPLE_API`
- `types.hpp` with `Roi`, `TrainOptions`, `TrainResult`, `MatchOptions`, `MatchHit`, `MatchResult`
- `shape_matcher.hpp` with the exported `ShapeMatcher` declaration, deleted copy operations, move support, and method signatures

Modify `CMakeLists.txt` to:

- build `shape_match_sample_lib` as `SHARED`
- define `SHAPE_MATCH_SAMPLE_BUILD_DLL` for the DLL target
- keep runtime DLL copying for CLI and tests

**Step 4: Run test to verify it passes compilation further**

Run: `cmake --build build --config Debug --target dll_train_match`

Expected: compile now proceeds into missing method definitions rather than missing headers or class declarations.

**Step 5: Commit**

```bash
git add CMakeLists.txt include/shape_match_sample/export.hpp include/shape_match_sample/types.hpp include/shape_match_sample/shape_matcher.hpp
git commit -m "feat: add exported shape matcher api"
```

### Task 3: Implement training, save, and load in the DLL

**Files:**
- Create: `src/shape_matcher.cpp`
- Modify: `src/pipeline.cpp`
- Modify: `src/main.cpp`
- Modify: `tests/dll_persistence.cpp`
- Modify: `tests/train_smoke.cpp`
- Modify: `tests/roi_train.cpp`

**Step 1: Write the failing test**

In `tests/dll_persistence.cpp`, assert a full roundtrip:

```cpp
matcher.train(templ);
matcher.save(model_dir);

shape_match_sample::ShapeMatcher loaded;
loaded.load(model_dir);

if (loaded.empty()) {
    return 1;
}
```

**Step 2: Run test to verify it fails**

Run: `cmake --build build --config Debug --target dll_persistence`

Expected: link failure or runtime failure because `train`, `save`, and `load` are not implemented yet.

**Step 3: Write minimal implementation**

Implement in-memory model ownership in `src/shape_matcher.cpp`:

- validate template image and ROI
- create detector and training infos
- keep metadata and infos in memory
- persist `template.yaml`, `info.yaml`, and `model_meta.yaml`
- load those files back into a fresh matcher

Update `src/main.cpp` so the `train` CLI path uses `ShapeMatcher::trainFromFile(...)` and `save(...)`.

Update the training tests to include `shape_match_sample/shape_matcher.hpp` and call the class API instead of the old free functions.

**Step 4: Run tests to verify they pass**

Run: `ctest --test-dir build -C Debug --output-on-failure -R "train_smoke|roi_train|dll_persistence"`

Expected: the selected training and persistence tests pass.

**Step 5: Commit**

```bash
git add src/shape_matcher.cpp src/pipeline.cpp src/main.cpp tests/dll_persistence.cpp tests/train_smoke.cpp tests/roi_train.cpp
git commit -m "feat: add dll training and persistence api"
```

### Task 4: Implement in-memory match and overlay rendering

**Files:**
- Modify: `src/shape_matcher.cpp`
- Modify: `src/main.cpp`
- Modify: `tests/dll_train_match.cpp`
- Modify: `tests/match_smoke.cpp`
- Modify: `tests/roi_match.cpp`

**Step 1: Write the failing test**

Extend `tests/dll_train_match.cpp` to check both `match(...)` and `renderMatches(...)`:

```cpp
const auto result = matcher.match(scene);
if (result.matches.empty()) {
    return 1;
}
const cv::Mat overlay = matcher.renderMatches(scene, result);
if (overlay.empty()) {
    return 1;
}
```

**Step 2: Run test to verify it fails**

Run: `cmake --build build --config Debug --target dll_train_match`

Expected: link failure or runtime failure because `match` or `renderMatches` is incomplete.

**Step 3: Write minimal implementation**

Implement:

- `match(const cv::Mat&, const MatchOptions&) const`
- `matchToFile(...) const`
- `matchFromFile(...) const`
- `renderMatches(...) const`

Update `src/main.cpp` so the `match` CLI path uses `load(...)` and `matchFromFile(...)`.

Update the matching tests to use the class API and preserve ROI assertions.

**Step 4: Run tests to verify they pass**

Run: `ctest --test-dir build -C Debug --output-on-failure -R "match_smoke|roi_match|dll_train_match|cli_match"`

Expected: the selected matching tests pass.

**Step 5: Commit**

```bash
git add src/shape_matcher.cpp src/main.cpp tests/dll_train_match.cpp tests/match_smoke.cpp tests/roi_match.cpp
git commit -m "feat: add dll matching and overlay api"
```

### Task 5: Refresh docs and run full verification

**Files:**
- Modify: `README.md`
- Modify: `CMakeLists.txt`

**Step 1: Write the failing test**

Add a README checklist for:

- DLL build output exists
- class API example exists
- CLI example still exists

Use this checklist while reviewing the rendered README after edits.

**Step 2: Run test to verify it fails**

Run: `rg -n "DLL|ShapeMatcher|cv::Mat" README.md`

Expected: missing or incomplete results before the README update.

**Step 3: Write minimal implementation**

Update the README to describe:

- the DLL target
- the exported `ShapeMatcher` class
- an in-memory `cv::Mat` example
- the retained CLI usage

If needed, make final CMake cleanup for installable target names or output clarity.

**Step 4: Run full verification**

Run:

```bash
cmake -S . -B build -DOpenCV_DIR=D:/proj/3rdParty/opencv410/build
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

Expected:

- configure succeeds
- build succeeds
- all tests pass

**Step 5: Commit**

```bash
git add README.md CMakeLists.txt
git commit -m "docs: document dll shape matcher api"
```
