#include "shape_match_sample/shape_matcher.hpp"

#include <filesystem>
#include <iostream>

#include <opencv2/imgcodecs.hpp>

int main() {
    const auto source_dir = std::filesystem::path{SHAPE_MATCH_SAMPLE_SOURCE_DIR};
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

    shape_match_sample::ShapeMatcher matcher;
    if (!matcher.empty()) {
        std::cerr << "Fresh matcher should be empty.\n";
        return 1;
    }

    matcher.train(template_image);
    const auto result = matcher.match(scene_image);
    if (result.matches.empty()) {
        std::cerr << "Expected at least one match.\n";
        return 1;
    }

    const cv::Mat overlay = matcher.renderMatches(scene_image, result);
    if (overlay.empty()) {
        std::cerr << "Overlay image should not be empty.\n";
        return 1;
    }

    return 0;
}
