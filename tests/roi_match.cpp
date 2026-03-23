#include "shape_match_sample/shape_matcher.hpp"

#include <filesystem>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

namespace {

shape_match_sample::Roi non_black_bbox(const std::filesystem::path& image_path) {
    const cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image for ROI match test.");
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
        throw std::runtime_error("No non-black pixels found.");
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
    const auto scene_path = source_dir / "assets" / "scene.bmp";
    const cv::Mat template_image = cv::imread(template_path.string(), cv::IMREAD_COLOR);
    const cv::Mat scene_image = cv::imread(scene_path.string(), cv::IMREAD_COLOR);
    if (template_image.empty() || scene_image.empty()) {
        std::cerr << "Failed to load ROI test images.\n";
        return 1;
    }

    shape_match_sample::TrainOptions train_options;
    train_options.train_roi = non_black_bbox(template_path);
    shape_match_sample::ShapeMatcher matcher;
    matcher.train(template_image, train_options);

    shape_match_sample::MatchOptions match_options;
    match_options.search_roi = non_black_bbox(scene_path);

    const auto match_result = matcher.match(scene_image, match_options);
    if (match_result.matches.empty()) {
        std::cerr << "No matches were produced for ROI matching.\n";
        return 1;
    }

    const auto roi = *match_options.search_roi;
    const auto& best = match_result.matches.front();
    if (best.x < roi.x || best.y < roi.y) {
        std::cerr << "Match coordinates were not mapped back to full image space.\n";
        return 1;
    }
    if (best.x + best.width > roi.x + roi.width || best.y + best.height > roi.y + roi.height) {
        std::cerr << "Match lies outside requested search ROI.\n";
        return 1;
    }
    return 0;
}
