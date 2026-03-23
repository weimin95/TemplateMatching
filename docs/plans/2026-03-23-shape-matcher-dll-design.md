# Shape Matcher DLL Design

This change turns the current `shape_based_matching` sample into a reusable C++ DLL while keeping the CLI as a thin wrapper around the library API.

## Goals

- Export a C++ class interface from a DLL.
- Use `cv::Mat` as the primary image type for external callers.
- Support both in-memory training/matching and persisted model save/load.
- Keep the existing CLI workflow available for manual testing and examples.

## Public API

The library will expose one primary class: `shape_match_sample::ShapeMatcher`.

The public API will include:

- `train(const cv::Mat&, const TrainOptions&)`
- `trainFromFile(const std::filesystem::path&, const TrainOptions&)`
- `save(const std::filesystem::path&) const`
- `load(const std::filesystem::path&, const std::string& class_id = "")`
- `match(const cv::Mat&, const MatchOptions&) const`
- `matchToFile(const cv::Mat&, const std::filesystem::path&, const MatchOptions&) const`
- `matchFromFile(const std::filesystem::path&, const std::filesystem::path&, const MatchOptions&) const`
- `renderMatches(const cv::Mat&, const MatchResult&) const`
- `empty() const noexcept`
- `clear() noexcept`

The interface will throw `std::runtime_error` for invalid input, missing model state, invalid ROI values, failed file I/O, and invalid persisted model files. Empty match results are not errors.

## Library Structure

The project will switch from a static helper library to a shared library target.

Public headers:

- `include/shape_match_sample/export.hpp`
- `include/shape_match_sample/types.hpp`
- `include/shape_match_sample/shape_matcher.hpp`

Internal implementation:

- `src/shape_matcher.cpp`
- helper logic reused from the current pipeline implementation for ROI validation, metadata I/O, padding, and overlay drawing

The DLL boundary will use a normal MSVC import/export macro:

- `__declspec(dllexport)` while building the DLL
- `__declspec(dllimport)` for external consumers

Because the user explicitly chose a direct C++ ABI, the exported interface may use `cv::Mat`, `std::string`, `std::vector`, and exceptions directly. This assumes the caller uses a compatible MSVC/OpenCV toolchain.

## Runtime State

`ShapeMatcher` will own the currently active model in memory through an internal implementation object.

The state will include:

- detector instance
- class id
- persisted model metadata
- loaded training info entries
- a flag that indicates whether a model is currently available

Training writes this state into memory first. `save()` persists that state to `template.yaml`, `info.yaml`, and `model_meta.yaml`. `load()` reconstructs the state from those files.

## Data Flow

Training:

1. Validate image and ROI.
2. Crop the training ROI.
3. Build a foreground mask from non-black pixels.
4. Generate rotated/scaled views through `shapeInfo_producer`.
5. Add valid templates to `line2Dup::Detector`.
6. Store detector, metadata, and training infos in memory.
7. Return `TrainResult`.

Matching:

1. Validate that a model is available.
2. Validate scene image and search ROI.
3. Crop and pad the search image.
4. Run `Detector::match(...)`.
5. Map ROI-local coordinates back to the full scene image.
6. Return `MatchResult`.
7. Optionally render and write an overlay image.

Rendering:

1. Copy the input scene image.
2. Draw the search ROI.
3. Draw hit rectangles, feature points, and similarity text.
4. Return the overlay `cv::Mat`.

## CLI Relationship

The CLI remains in the repository, but it becomes a thin wrapper over `ShapeMatcher`.

- `train` uses `trainFromFile(...)` and `save(...)`
- `match` uses `load(...)`, `matchFromFile(...)`, and the existing console output format

This keeps the current smoke-test workflow intact while moving the actual implementation into the DLL.

## Testing

The test suite will be extended to cover:

- in-memory train and match using `cv::Mat`
- save/load roundtrip through the DLL API
- ROI training and ROI matching through the DLL API
- existing CLI train and match flows using the DLL-backed implementation

The shared library target will be linked by both the CLI and the tests, and runtime DLL copying will stay enabled on Windows so local test execution still works.
