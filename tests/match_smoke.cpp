#include "shape_match_sample/shape_matcher.hpp"

#include <filesystem>
#include <iostream>

#include <opencv2/imgcodecs.hpp>

int main() {
    const auto source_dir = std::filesystem::path{SHAPE_MATCH_SAMPLE_SOURCE_DIR};
    const auto template_path = source_dir / "assets" / "template.bmp";
    const auto scene_path = source_dir / "assets" / "scene.bmp";
    const auto workspace = std::filesystem::current_path() / "match_smoke_artifacts";
    const auto overlay_path = workspace / "output" / "match_overlay.png";

    const cv::Mat template_image = cv::imread(template_path.string(), cv::IMREAD_COLOR);
    const cv::Mat scene_image = cv::imread(scene_path.string(), cv::IMREAD_COLOR);
    if (template_image.empty() || scene_image.empty()) {
        std::cerr << "Failed to load test images.\n";
        return 1;
    }

    shape_match_sample::ShapeMatcher matcher;
    const auto train_result = matcher.train(template_image);
    if (train_result.template_count == 0) {
        std::cerr << "Training did not produce templates.\n";
        return 1;
    }

    const auto match_result = matcher.matchToFile(scene_image, overlay_path);
    if (match_result.matches.empty()) {
        std::cerr << "No matches were produced.\n";
        return 1;
    }
    if (!std::filesystem::exists(overlay_path)) {
        std::cerr << "Overlay image was not created.\n";
        return 1;
    }
    return 0;
}
