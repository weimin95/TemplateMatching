#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>

#include "shape_match_sample/export.hpp"
#include "shape_match_sample/types.hpp"

namespace shape_match_sample {

class SHAPE_MATCH_SAMPLE_API ShapeMatcher {
public:
    ShapeMatcher();
    ~ShapeMatcher();

    ShapeMatcher(const ShapeMatcher&) = delete;
    ShapeMatcher& operator=(const ShapeMatcher&) = delete;
    ShapeMatcher(ShapeMatcher&& other) noexcept;
    ShapeMatcher& operator=(ShapeMatcher&& other) noexcept;

    TrainResult train(
        const cv::Mat& template_image,
        const TrainOptions& options = {});
    TrainResult trainFromFile(
        const std::filesystem::path& template_image_path,
        const TrainOptions& options = {});

    void save(const std::filesystem::path& model_dir) const;
    void load(
        const std::filesystem::path& model_dir,
        const std::string& class_id = "");

    MatchResult match(
        const cv::Mat& scene_image,
        const MatchOptions& options = {}) const;
    MatchResult matchToFile(
        const cv::Mat& scene_image,
        const std::filesystem::path& overlay_path,
        const MatchOptions& options = {}) const;
    MatchResult matchFromFile(
        const std::filesystem::path& scene_image_path,
        const std::filesystem::path& overlay_path,
        const MatchOptions& options = {}) const;

    cv::Mat renderMatches(
        const cv::Mat& scene_image,
        const MatchResult& result) const;

    bool empty() const noexcept;
    void clear() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace shape_match_sample
