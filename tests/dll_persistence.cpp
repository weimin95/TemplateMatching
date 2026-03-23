#include "shape_match_sample/shape_matcher.hpp"

#include <filesystem>
#include <iostream>

#include <opencv2/imgcodecs.hpp>

int main() {
    const auto source_dir = std::filesystem::path{SHAPE_MATCH_SAMPLE_SOURCE_DIR};
    const auto workspace = std::filesystem::current_path() / "dll_persistence_artifacts";
    const auto model_dir = workspace / "models" / "default";

    const cv::Mat template_image = cv::imread(
        (source_dir / "assets" / "template.bmp").string(),
        cv::IMREAD_COLOR);
    const cv::Mat scene_image = cv::imread(
        (source_dir / "assets" / "scene.bmp").string(),
        cv::IMREAD_COLOR);

    if (template_image.empty() || scene_image.empty()) {
        std::cerr << "Failed to load test assets.\n";
        return 1;
    }

    shape_match_sample::ShapeMatcher trained;
    trained.train(template_image);
    trained.save(model_dir);

    shape_match_sample::ShapeMatcher loaded;
    loaded.load(model_dir);
    if (loaded.empty()) {
        std::cerr << "Loaded matcher should not be empty.\n";
        return 1;
    }

    const auto result = loaded.match(scene_image);
    if (result.matches.empty()) {
        std::cerr << "Expected at least one persisted match.\n";
        return 1;
    }

    return 0;
}
