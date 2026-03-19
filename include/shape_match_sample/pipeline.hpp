#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace shape_match_sample {

struct Roi {
    int x{};
    int y{};
    int width{};
    int height{};
};

struct TrainOptions {
    std::string class_id{"default"};
    std::optional<Roi> train_roi;
    int num_features{128};
    std::vector<int> pyramid_levels{4, 8};
    float weak_threshold{30.0f};
    float strong_threshold{60.0f};
    float angle_start{0.0f};
    float angle_end{0.0f};
    float angle_step{1.0f};
    float scale_start{1.0f};
    float scale_end{1.0f};
    float scale_step{0.1f};
};

struct TrainResult {
    std::size_t generated_view_count{};
    std::size_t template_count{};
    Roi effective_train_roi{};
    std::filesystem::path template_yaml_path;
    std::filesystem::path info_yaml_path;
    std::filesystem::path meta_yaml_path;
};

struct MatchOptions {
    std::string class_id{"default"};
    std::optional<Roi> search_roi;
    float min_score{90.0f};
    std::size_t top_k{5};
};

struct MatchHit {
    int x{};
    int y{};
    int width{};
    int height{};
    float similarity{};
    int template_id{};
};

struct MatchResult {
    Roi effective_search_roi{};
    std::vector<MatchHit> matches;
    std::filesystem::path overlay_path;
};

TrainResult train_model(
    const std::filesystem::path& template_image_path,
    const std::filesystem::path& model_dir,
    const TrainOptions& options = {});

MatchResult match_model(
    const std::filesystem::path& scene_image_path,
    const std::filesystem::path& model_dir,
    const std::filesystem::path& overlay_path,
    const MatchOptions& options = {});

}  // namespace shape_match_sample
