#include "shape_match_sample/pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "line2Dup.h"

namespace shape_match_sample {
namespace {

struct ModelMetadata {
    std::string class_id;
    Roi train_roi;
    int template_width{};
    int template_height{};
    int num_features{};
    std::vector<int> pyramid_levels;
    float weak_threshold{};
    float strong_threshold{};
};

cv::Rect resolve_roi(const std::optional<Roi>& roi, const cv::Size& image_size, const char* label) {
    if (!roi.has_value()) {
        return cv::Rect{0, 0, image_size.width, image_size.height};
    }

    if (roi->width <= 0 || roi->height <= 0) {
        throw std::runtime_error(std::string(label) + " ROI width and height must be positive.");
    }
    if (roi->x < 0 || roi->y < 0) {
        throw std::runtime_error(std::string(label) + " ROI x and y must be non-negative.");
    }
    if (roi->x + roi->width > image_size.width || roi->y + roi->height > image_size.height) {
        throw std::runtime_error(std::string(label) + " ROI exceeds image bounds.");
    }

    return cv::Rect{roi->x, roi->y, roi->width, roi->height};
}

Roi to_roi(const cv::Rect& rect) {
    return Roi{rect.x, rect.y, rect.width, rect.height};
}

cv::Mat make_foreground_mask(const cv::Mat& image) {
    cv::Mat grayscale;
    if (image.channels() == 1) {
        grayscale = image;
    } else {
        cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
    }

    cv::Mat mask;
    cv::threshold(grayscale, mask, 0.0, 255.0, cv::THRESH_BINARY);
    return mask;
}

void assign_shape_ranges(
    shape_based_matching::shapeInfo_producer& shapes,
    const TrainOptions& options) {
    if (options.angle_end > options.angle_start) {
        shapes.angle_range = {options.angle_start, options.angle_end};
    } else {
        shapes.angle_range = {options.angle_start};
    }
    shapes.angle_step = options.angle_step;

    if (options.scale_end > options.scale_start) {
        shapes.scale_range = {options.scale_start, options.scale_end};
    } else {
        shapes.scale_range = {options.scale_start};
    }
    shapes.scale_step = options.scale_step;
}

void write_model_metadata(
    const std::filesystem::path& meta_path,
    const TrainOptions& options,
    const cv::Size& template_size,
    const Roi& train_roi,
    std::size_t generated_view_count,
    std::size_t template_count) {
    cv::FileStorage fs(meta_path.string(), cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open metadata file for writing: " + meta_path.string());
    }

    fs << "class_id" << options.class_id;
    fs << "template_width" << template_size.width;
    fs << "template_height" << template_size.height;

    fs << "train_roi" << "{";
    fs << "x" << train_roi.x;
    fs << "y" << train_roi.y;
    fs << "width" << train_roi.width;
    fs << "height" << train_roi.height;
    fs << "}";

    fs << "detector" << "{";
    fs << "num_features" << options.num_features;
    fs << "weak_threshold" << options.weak_threshold;
    fs << "strong_threshold" << options.strong_threshold;
    fs << "pyramid_levels" << "[";
    for (const auto level : options.pyramid_levels) {
        fs << level;
    }
    fs << "]";
    fs << "}";

    fs << "augment" << "{";
    fs << "angle_start" << options.angle_start;
    fs << "angle_end" << options.angle_end;
    fs << "angle_step" << options.angle_step;
    fs << "scale_start" << options.scale_start;
    fs << "scale_end" << options.scale_end;
    fs << "scale_step" << options.scale_step;
    fs << "}";

    fs << "generated_view_count" << static_cast<int>(generated_view_count);
    fs << "template_count" << static_cast<int>(template_count);
}

ModelMetadata load_model_metadata(const std::filesystem::path& meta_path) {
    cv::FileStorage fs(meta_path.string(), cv::FileStorage::READ);
    if (!fs.isOpened()) {
        throw std::runtime_error("Failed to open model metadata: " + meta_path.string());
    }

    ModelMetadata meta;
    meta.class_id = static_cast<std::string>(fs["class_id"]);
    meta.template_width = static_cast<int>(fs["template_width"]);
    meta.template_height = static_cast<int>(fs["template_height"]);

    const cv::FileNode train_roi = fs["train_roi"];
    meta.train_roi.x = static_cast<int>(train_roi["x"]);
    meta.train_roi.y = static_cast<int>(train_roi["y"]);
    meta.train_roi.width = static_cast<int>(train_roi["width"]);
    meta.train_roi.height = static_cast<int>(train_roi["height"]);

    const cv::FileNode detector = fs["detector"];
    meta.num_features = static_cast<int>(detector["num_features"]);
    meta.weak_threshold = static_cast<float>(detector["weak_threshold"]);
    meta.strong_threshold = static_cast<float>(detector["strong_threshold"]);

    const cv::FileNode pyramid_levels = detector["pyramid_levels"];
    for (const auto& node : pyramid_levels) {
        meta.pyramid_levels.push_back(static_cast<int>(node));
    }

    if (meta.class_id.empty()) {
        throw std::runtime_error("Model metadata is missing class_id.");
    }
    if (meta.pyramid_levels.empty()) {
        throw std::runtime_error("Model metadata is missing pyramid levels.");
    }

    return meta;
}

void draw_match_overlay(
    cv::Mat& overlay,
    const std::vector<MatchHit>& hits,
    const line2Dup::Detector& detector,
    const std::string& class_id) {
    static const cv::Scalar colors[] = {
        {0, 255, 255},
        {0, 200, 0},
        {255, 160, 0},
        {255, 0, 255},
        {0, 128, 255}
    };

    for (std::size_t i = 0; i < hits.size(); ++i) {
        const auto& hit = hits[i];
        const auto color = colors[i % (sizeof(colors) / sizeof(colors[0]))];
        const auto& templ = detector.getTemplates(class_id, hit.template_id);

        cv::rectangle(
            overlay,
            cv::Rect{hit.x, hit.y, hit.width, hit.height},
            color,
            2,
            cv::LINE_AA);

        for (const auto& feature : templ[0].features) {
            cv::circle(
                overlay,
                cv::Point{hit.x + feature.x, hit.y + feature.y},
                2,
                color,
                -1,
                cv::LINE_AA);
        }

        cv::putText(
            overlay,
            std::to_string(static_cast<int>(std::lround(hit.similarity))),
            cv::Point{hit.x, std::max(18, hit.y - 6)},
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv::LINE_AA);
    }
}

int required_match_stride(const std::vector<int>& pyramid_levels) {
    int stride = 1;
    for (const auto level : pyramid_levels) {
        stride = std::max(stride, level);
    }

    int pyramid_scale = 1;
    for (std::size_t i = 1; i < pyramid_levels.size(); ++i) {
        pyramid_scale *= 2;
    }
    return stride * pyramid_scale;
}

cv::Mat pad_for_matching(const cv::Mat& image, int stride) {
    const int padded_width = ((image.cols + stride - 1) / stride) * stride;
    const int padded_height = ((image.rows + stride - 1) / stride) * stride;

    if (padded_width == image.cols && padded_height == image.rows) {
        return image;
    }

    cv::Mat padded;
    cv::copyMakeBorder(
        image,
        padded,
        0,
        padded_height - image.rows,
        0,
        padded_width - image.cols,
        cv::BORDER_CONSTANT,
        cv::Scalar::all(0));
    return padded;
}

}  // namespace

TrainResult train_model(
    const std::filesystem::path& template_image_path,
    const std::filesystem::path& model_dir,
    const TrainOptions& options) {
    if (options.pyramid_levels.empty()) {
        throw std::runtime_error("At least one pyramid level is required.");
    }

    const cv::Mat template_image = cv::imread(template_image_path.string(), cv::IMREAD_COLOR);
    if (template_image.empty()) {
        throw std::runtime_error("Failed to read template image: " + template_image_path.string());
    }

    const cv::Rect train_rect = resolve_roi(options.train_roi, template_image.size(), "Training");
    const cv::Mat train_crop = template_image(train_rect).clone();
    cv::Mat mask = make_foreground_mask(train_crop);
    if (cv::countNonZero(mask) == 0) {
        throw std::runtime_error("Training ROI does not contain any non-black pixels.");
    }

    line2Dup::Detector detector(
        options.num_features,
        options.pyramid_levels,
        options.weak_threshold,
        options.strong_threshold);

    shape_based_matching::shapeInfo_producer shapes(train_crop, mask);
    assign_shape_ranges(shapes, options);
    shapes.produce_infos();

    std::vector<shape_based_matching::shapeInfo_producer::Info> infos_with_templates;
    infos_with_templates.reserve(shapes.infos.size());

    for (const auto& info : shapes.infos) {
        const int template_id = detector.addTemplate(
            shapes.src_of(info),
            options.class_id,
            shapes.mask_of(info),
            options.num_features);
        if (template_id != -1) {
            infos_with_templates.push_back(info);
        }
    }

    if (infos_with_templates.empty()) {
        throw std::runtime_error("Training did not produce any valid templates.");
    }

    std::filesystem::create_directories(model_dir);
    const auto template_yaml_path = model_dir / "template.yaml";
    const auto info_yaml_path = model_dir / "info.yaml";
    const auto meta_yaml_path = model_dir / "model_meta.yaml";

    cv::FileStorage template_fs(template_yaml_path.string(), cv::FileStorage::WRITE);
    if (!template_fs.isOpened()) {
        throw std::runtime_error("Failed to open template.yaml for writing: " + template_yaml_path.string());
    }
    detector.writeClass(options.class_id, template_fs);

    shape_based_matching::shapeInfo_producer::save_infos(infos_with_templates, info_yaml_path.string());
    write_model_metadata(
        meta_yaml_path,
        options,
        template_image.size(),
        to_roi(train_rect),
        shapes.infos.size(),
        infos_with_templates.size());

    return TrainResult{
        infos_with_templates.size(),
        infos_with_templates.size(),
        to_roi(train_rect),
        template_yaml_path,
        info_yaml_path,
        meta_yaml_path
    };
}

MatchResult match_model(
    const std::filesystem::path& scene_image_path,
    const std::filesystem::path& model_dir,
    const std::filesystem::path& overlay_path,
    const MatchOptions& options) {
    const cv::Mat scene_image = cv::imread(scene_image_path.string(), cv::IMREAD_COLOR);
    if (scene_image.empty()) {
        throw std::runtime_error("Failed to read scene image: " + scene_image_path.string());
    }

    const auto template_yaml_path = model_dir / "template.yaml";
    const auto info_yaml_path = model_dir / "info.yaml";
    const auto meta_yaml_path = model_dir / "model_meta.yaml";
    if (!std::filesystem::exists(template_yaml_path)) {
        throw std::runtime_error("template.yaml was not found: " + template_yaml_path.string());
    }
    if (!std::filesystem::exists(info_yaml_path)) {
        throw std::runtime_error("info.yaml was not found: " + info_yaml_path.string());
    }
    if (!std::filesystem::exists(meta_yaml_path)) {
        throw std::runtime_error("model_meta.yaml was not found: " + meta_yaml_path.string());
    }

    const auto meta = load_model_metadata(meta_yaml_path);
    const auto class_id = options.class_id.empty() ? meta.class_id : options.class_id;

    line2Dup::Detector detector(
        meta.num_features,
        meta.pyramid_levels,
        meta.weak_threshold,
        meta.strong_threshold);

    cv::FileStorage template_fs(template_yaml_path.string(), cv::FileStorage::READ);
    if (!template_fs.isOpened()) {
        throw std::runtime_error("Failed to open template.yaml for reading: " + template_yaml_path.string());
    }
    detector.readClass(template_fs.root(), class_id);

    const auto infos = shape_based_matching::shapeInfo_producer::load_infos(info_yaml_path.string());
    (void)infos;

    const cv::Rect search_rect = resolve_roi(options.search_roi, scene_image.size(), "Search");
    const cv::Mat search_crop = scene_image(search_rect).clone();
    const int stride = required_match_stride(meta.pyramid_levels);
    const cv::Mat padded_search_crop = pad_for_matching(search_crop, stride);

    const auto raw_matches = detector.match(padded_search_crop, options.min_score, {class_id});

    cv::Mat overlay = scene_image.clone();
    cv::rectangle(overlay, search_rect, cv::Scalar{255, 255, 0}, 2, cv::LINE_AA);

    std::vector<MatchHit> hits;
    const auto max_hits = std::min(options.top_k, raw_matches.size());
    hits.reserve(max_hits);

    for (std::size_t i = 0; i < max_hits; ++i) {
        const auto& match = raw_matches[i];
        const auto& templ = detector.getTemplates(class_id, match.template_id);

        hits.push_back(MatchHit{
            search_rect.x + match.x,
            search_rect.y + match.y,
            templ[0].width,
            templ[0].height,
            match.similarity,
            match.template_id
        });
    }

    if (hits.empty()) {
        cv::putText(
            overlay,
            "0 matches",
            cv::Point{12, 28},
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar{0, 0, 255},
            2,
            cv::LINE_AA);
    } else {
        draw_match_overlay(overlay, hits, detector, class_id);
    }

    std::filesystem::create_directories(overlay_path.parent_path());
    if (!cv::imwrite(overlay_path.string(), overlay)) {
        throw std::runtime_error("Failed to write overlay image: " + overlay_path.string());
    }

    return MatchResult{
        to_roi(search_rect),
        std::move(hits),
        overlay_path
    };
}

}  // namespace shape_match_sample
