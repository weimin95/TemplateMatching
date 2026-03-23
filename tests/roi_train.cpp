#include "shape_match_sample/shape_matcher.hpp"

#include <filesystem>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace {

shape_match_sample::Roi non_black_bbox(const std::filesystem::path& image_path) {
    const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load template image for ROI test.");
    }

    int min_x = image.cols;
    int min_y = image.rows;
    int max_x = -1;
    int max_y = -1;

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            const auto pixel = image.at<cv::Vec3b>(y, x);
            if (pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0) {
                min_x = std::min(min_x, x);
                min_y = std::min(min_y, y);
                max_x = std::max(max_x, x);
                max_y = std::max(max_y, y);
            }
        }
    }

    if (max_x < min_x || max_y < min_y) {
        throw std::runtime_error("No non-black pixels found in template image.");
    }

    return {
        min_x,
        min_y,
        max_x - min_x + 1,
        max_y - min_y + 1
    };
}

}  // namespace

int main() {
    const auto source_dir = std::filesystem::path{SHAPE_MATCH_SAMPLE_SOURCE_DIR};
    const auto template_path = source_dir / "assets" / "template.bmp";
    const auto model_dir = std::filesystem::current_path() / "roi_train_artifacts" / "models" / "default";
    const cv::Mat template_image = cv::imread(template_path.string(), cv::IMREAD_COLOR);
    if (template_image.empty()) {
        std::cerr << "Failed to load template image.\n";
        return 1;
    }

    shape_match_sample::TrainOptions options;
    options.train_roi = non_black_bbox(template_path);

    shape_match_sample::ShapeMatcher matcher;
    const auto result = matcher.train(template_image, options);
    matcher.save(model_dir);

    if (result.effective_train_roi.width != options.train_roi->width ||
        result.effective_train_roi.height != options.train_roi->height) {
        std::cerr << "Returned ROI does not match requested ROI.\n";
        return 1;
    }
    if (!std::filesystem::exists(model_dir / "model_meta.yaml")) {
        std::cerr << "model_meta.yaml was not created.\n";
        return 1;
    }
    return 0;
}
