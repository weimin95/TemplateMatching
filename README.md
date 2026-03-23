# Shape Matcher DLL Sample

This repository is a clean rebuild around `https://github.com/meiqua/shape_based_matching`.

It uses:

- an exported C++ DLL API built around `ShapeMatcher`
- OpenCV for image loading and visualization
- upstream `line2Dup` from `shape_based_matching`
- a retained CLI wrapper for `train` and `match`

The sample assets are:

- `assets/template.bmp`
- `assets/scene.bmp`

## Prerequisites

- CMake 3.25 or newer
- A C++17 compiler
- OpenCV installed and discoverable by CMake

If CMake cannot find OpenCV automatically, configure with `-DOpenCV_DIR=...`.

## Configure and Build

```bash
cmake -S . -B build -DOpenCV_DIR=D:/proj/3rdParty/opencv410/build
cmake --build build --config Debug
```

The main outputs are:

- `build/Debug/shape_matcher.dll`
- `build/Debug/shape_matcher.lib`
- `build/Debug/shape_match_sample.exe`

## DLL API

Public headers:

- `include/shape_match_sample/shape_matcher.hpp`
- `include/shape_match_sample/types.hpp`

Minimal in-memory usage:

```cpp
#include "shape_match_sample/shape_matcher.hpp"

#include <opencv2/imgcodecs.hpp>

int main() {
    const cv::Mat template_image = cv::imread("assets/template.bmp", cv::IMREAD_COLOR);
    const cv::Mat scene_image = cv::imread("assets/scene.bmp", cv::IMREAD_COLOR);

    shape_match_sample::ShapeMatcher matcher;
    matcher.train(template_image);
    matcher.save("models/default");

    shape_match_sample::ShapeMatcher loaded;
    loaded.load("models/default");

    const auto result = loaded.match(scene_image);
    const cv::Mat overlay = loaded.renderMatches(scene_image, result);
    cv::imwrite("output/match_overlay.png", overlay);
    return 0;
}
```

Main API surface:

- `train(const cv::Mat&, const TrainOptions&)`
- `save(const std::filesystem::path&)`
- `load(const std::filesystem::path&, const std::string& class_id = "")`
- `match(const cv::Mat&, const MatchOptions&) const`
- `renderMatches(const cv::Mat&, const MatchResult&) const`

## CLI Wrapper

The CLI is still available, but it now calls into the DLL API.

### Train

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

### Match

```bash
build/Debug/shape_match_sample match assets/scene.bmp models/default output/match_overlay.png
```

Match with an explicit search ROI:

```bash
build/Debug/shape_match_sample match assets/scene.bmp models/default output/match_overlay.png --search-roi x,y,w,h
```

The overlay image is written to the requested path. If no overlay path is provided, the default is `output/match_overlay.png`.

## ROI Rules

- `TrainOptions::train_roi` and `MatchOptions::search_roi` use `x,y,w,h`
- CLI flags `--train-roi` and `--search-roi` use the same format
- both are optional
- if omitted, the whole image is used
- invalid ROI values fail immediately

## Test

```bash
ctest --test-dir build -C Debug --output-on-failure
```
