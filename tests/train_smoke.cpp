#include "shape_match_sample/pipeline.hpp"

#include <filesystem>
#include <iostream>

int main() {
    const auto source_dir = std::filesystem::path{SHAPE_MATCH_SAMPLE_SOURCE_DIR};
    const auto template_path = source_dir / "assets" / "template.bmp";
    const auto model_dir = std::filesystem::current_path() / "train_smoke_artifacts" / "models" / "default";

    const auto result = shape_match_sample::train_model(template_path, model_dir);

    if (result.template_count == 0) {
        std::cerr << "No templates were generated.\n";
        return 1;
    }
    if (!std::filesystem::exists(result.template_yaml_path)) {
        std::cerr << "template.yaml was not created.\n";
        return 1;
    }
    if (!std::filesystem::exists(result.info_yaml_path)) {
        std::cerr << "info.yaml was not created.\n";
        return 1;
    }
    if (!std::filesystem::exists(result.meta_yaml_path)) {
        std::cerr << "model_meta.yaml was not created.\n";
        return 1;
    }
    return 0;
}
