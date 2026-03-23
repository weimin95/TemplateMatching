#include "shape_match_sample/shape_matcher.hpp"

#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct ParsedArgs {
    std::vector<std::string> positionals;
    std::optional<shape_match_sample::Roi> train_roi;
    std::optional<shape_match_sample::Roi> search_roi;
};

bool looks_like_path(const std::string& value) {
    return value.find('/') != std::string::npos ||
           value.find('\\') != std::string::npos ||
           value.find('.') != std::string::npos;
}

shape_match_sample::Roi parse_roi(const std::string& text) {
    std::vector<int> values;
    std::string current;

    for (const auto ch : text) {
        if (ch == ',') {
            if (current.empty()) {
                throw std::runtime_error("ROI contains an empty field.");
            }
            values.push_back(std::stoi(current));
            current.clear();
        } else {
            current.push_back(ch);
        }
    }

    if (!current.empty()) {
        values.push_back(std::stoi(current));
    }
    if (values.size() != 4) {
        throw std::runtime_error("ROI must use x,y,w,h format.");
    }

    return shape_match_sample::Roi{values[0], values[1], values[2], values[3]};
}

ParsedArgs parse_command_args(int argc, char** argv, int start_index) {
    ParsedArgs parsed;

    for (int i = start_index; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--train-roi") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--train-roi requires a value.");
            }
            parsed.train_roi = parse_roi(argv[++i]);
        } else if (arg == "--search-roi") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--search-roi requires a value.");
            }
            parsed.search_roi = parse_roi(argv[++i]);
        } else {
            parsed.positionals.push_back(arg);
        }
    }

    return parsed;
}

void print_usage(std::string_view exe_name) {
    std::cerr
        << "Usage:\n"
        << "  " << exe_name << " train <template_image> <model_dir> [class_id] [--train-roi x,y,w,h]\n"
        << "  " << exe_name << " match <scene_image> <model_dir> [class_id] [overlay_output] [--search-roi x,y,w,h]\n";
}

int run_train(const ParsedArgs& parsed) {
    if (parsed.positionals.size() < 2 || parsed.positionals.size() > 3) {
        throw std::runtime_error("train expects 2 or 3 positional arguments.");
    }

    shape_match_sample::TrainOptions options;
    if (parsed.positionals.size() == 3) {
        options.class_id = parsed.positionals[2];
    }
    options.train_roi = parsed.train_roi;

    shape_match_sample::ShapeMatcher matcher;
    const auto result = matcher.trainFromFile(parsed.positionals[0], options);
    const std::filesystem::path model_dir = parsed.positionals[1];
    matcher.save(model_dir);

    std::cout
        << "Generated views: " << result.generated_view_count << '\n'
        << "Template count: " << result.template_count << '\n'
        << "Train ROI: " << result.effective_train_roi.x << ','
        << result.effective_train_roi.y << ','
        << result.effective_train_roi.width << ','
        << result.effective_train_roi.height << '\n'
        << "template.yaml: " << std::filesystem::absolute(model_dir / "template.yaml").string() << '\n'
        << "info.yaml: " << std::filesystem::absolute(model_dir / "info.yaml").string() << '\n'
        << "model_meta.yaml: " << std::filesystem::absolute(model_dir / "model_meta.yaml").string() << '\n';
    return 0;
}

int run_match(const ParsedArgs& parsed) {
    if (parsed.positionals.size() < 2 || parsed.positionals.size() > 4) {
        throw std::runtime_error("match expects 2 to 4 positional arguments.");
    }

    shape_match_sample::MatchOptions options;
    options.search_roi = parsed.search_roi;

    std::filesystem::path overlay_path = std::filesystem::current_path() / "output" / "match_overlay.png";
    if (parsed.positionals.size() == 3) {
        if (looks_like_path(parsed.positionals[2])) {
            overlay_path = parsed.positionals[2];
        } else {
            options.class_id = parsed.positionals[2];
        }
    } else if (parsed.positionals.size() == 4) {
        options.class_id = parsed.positionals[2];
        overlay_path = parsed.positionals[3];
    }

    shape_match_sample::ShapeMatcher matcher;
    matcher.load(parsed.positionals[1], options.class_id);

    shape_match_sample::MatchOptions match_options = options;
    match_options.class_id.clear();
    const auto result = matcher.matchFromFile(
        parsed.positionals[0],
        overlay_path,
        match_options);

    std::cout
        << "Search ROI: " << result.effective_search_roi.x << ','
        << result.effective_search_roi.y << ','
        << result.effective_search_roi.width << ','
        << result.effective_search_roi.height << '\n'
        << "Match count: " << result.matches.size() << '\n';
    if (!result.matches.empty()) {
        std::cout << "Top similarity: " << result.matches.front().similarity << '\n';
    }
    std::cout << "Overlay: " << std::filesystem::absolute(result.overlay_path).string() << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage(argv[0]);
            return 1;
        }

        const std::string command = argv[1];
        if (command == "--help" || command == "-h") {
            print_usage(argv[0]);
            return 0;
        }

        const auto parsed = parse_command_args(argc, argv, 2);
        if (command == "train") {
            return run_train(parsed);
        }
        if (command == "match") {
            return run_match(parsed);
        }

        print_usage(argv[0]);
        return 1;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
