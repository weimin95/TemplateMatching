#include "shape_match_sample/pipeline.hpp"

#include <filesystem>
#include <iostream>

int main() {
    const auto source_dir = std::filesystem::path{SHAPE_MATCH_SAMPLE_SOURCE_DIR};
    const auto template_path = source_dir / "assets" / "template.bmp";
    const auto scene_path = source_dir / "assets" / "scene.bmp";
    const auto workspace = std::filesystem::current_path() / "match_smoke_artifacts";
    const auto model_dir = workspace / "models" / "default";
    const auto overlay_path = workspace / "output" / "match_overlay.png";

    shape_match_sample::TrainOptions train_options;
    train_options.class_id = "default";
    const auto train_result = shape_match_sample::train_model(template_path, model_dir, train_options);
    if (train_result.template_count == 0) {
        std::cerr << "Training did not produce templates.\n";
        return 1;
    }

    const auto match_result = shape_match_sample::match_model(scene_path, model_dir, overlay_path);
    if (match_result.matches.empty()) {
        std::cerr << "No matches were produced.\n";
        return 1;
    }
    if (!std::filesystem::exists(match_result.overlay_path)) {
        std::cerr << "Overlay image was not created.\n";
        return 1;
    }
    return 0;
}
