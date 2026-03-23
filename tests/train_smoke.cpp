#include "shape_match_sample/shape_matcher.hpp"

#include <filesystem>
#include <iostream>

#include <opencv2/imgcodecs.hpp>

int main() {
    const auto source_dir = std::filesystem::path{SHAPE_MATCH_SAMPLE_SOURCE_DIR};
    const auto template_path = source_dir / "assets" / "template.bmp";
    const auto model_dir = std::filesystem::current_path() / "train_smoke_artifacts" / "models" / "default";

    const cv::Mat template_image = cv::imread(template_path.string(), cv::IMREAD_COLOR);
    if (template_image.empty()) {
        std::cerr << "Failed to load template image.\n";
        return 1;
    }

    shape_match_sample::ShapeMatcher matcher;
    const auto result = matcher.train(template_image);
    matcher.save(model_dir);

    if (result.template_count == 0) {
        std::cerr << "No templates were generated.\n";
        return 1;
    }
    if (!std::filesystem::exists(model_dir / "template.yaml")) {
        std::cerr << "template.yaml was not created.\n";
        return 1;
    }
    if (!std::filesystem::exists(model_dir / "info.yaml")) {
        std::cerr << "info.yaml was not created.\n";
        return 1;
    }
    if (!std::filesystem::exists(model_dir / "model_meta.yaml")) {
        std::cerr << "model_meta.yaml was not created.\n";
        return 1;
    }
    return 0;
}
