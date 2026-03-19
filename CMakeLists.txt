cmake_minimum_required(VERSION 3.25)

if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()

project(shape_match_sample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui)

FetchContent_Declare(
    shape_based_matching_src
    GIT_REPOSITORY https://github.com/meiqua/shape_based_matching.git
    GIT_TAG master
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)

FetchContent_GetProperties(shape_based_matching_src)
if(NOT shape_based_matching_src_POPULATED)
    FetchContent_Populate(shape_based_matching_src)
endif()

set(SHAPE_MATCH_OPENCV_TARGETS)
if(TARGET OpenCV::opencv_core)
    list(APPEND SHAPE_MATCH_OPENCV_TARGETS
        OpenCV::opencv_core
        OpenCV::opencv_imgcodecs
        OpenCV::opencv_imgproc
        OpenCV::opencv_highgui
    )
elseif(TARGET opencv_core)
    list(APPEND SHAPE_MATCH_OPENCV_TARGETS
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        opencv_highgui
    )
else()
    list(APPEND SHAPE_MATCH_OPENCV_TARGETS ${OpenCV_LIBS})
endif()

add_library(shape_based_matching_upstream STATIC
    ${shape_based_matching_src_SOURCE_DIR}/line2Dup.cpp
)
target_include_directories(shape_based_matching_upstream PUBLIC
    ${shape_based_matching_src_SOURCE_DIR}
    ${shape_based_matching_src_SOURCE_DIR}/MIPP
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(shape_based_matching_upstream PUBLIC ${SHAPE_MATCH_OPENCV_TARGETS})
if(MSVC)
    target_compile_options(shape_based_matching_upstream PRIVATE /bigobj)
endif()

enable_testing()

add_library(shape_match_sample_headers INTERFACE)
target_include_directories(shape_match_sample_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(shape_match_sample_headers INTERFACE
    SHAPE_MATCH_SAMPLE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

add_library(shape_match_sample_lib STATIC
    src/pipeline.cpp
)
target_link_libraries(shape_match_sample_lib PUBLIC
    shape_match_sample_headers
    shape_based_matching_upstream
    ${SHAPE_MATCH_OPENCV_TARGETS}
)
if(MSVC)
    target_compile_options(shape_match_sample_lib PRIVATE /bigobj)
endif()

add_executable(shape_match_sample
    src/main.cpp
)
target_link_libraries(shape_match_sample PRIVATE
    shape_match_sample_lib
)

function(shape_match_copy_runtime_dlls target_name)
    if(WIN32)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${target_name}>
                $<TARGET_FILE_DIR:${target_name}>
            COMMAND_EXPAND_LISTS
        )
    endif()
endfunction()

add_executable(roi_train
    tests/roi_train.cpp
)
target_link_libraries(roi_train PRIVATE
    shape_match_sample_lib
)

add_executable(match_smoke
    tests/match_smoke.cpp
)
target_link_libraries(match_smoke PRIVATE
    shape_match_sample_lib
)

add_executable(roi_match
    tests/roi_match.cpp
)
target_link_libraries(roi_match PRIVATE
    shape_match_sample_lib
)

add_executable(train_smoke
    tests/train_smoke.cpp
)
target_link_libraries(train_smoke PRIVATE
    shape_match_sample_lib
)

shape_match_copy_runtime_dlls(match_smoke)
shape_match_copy_runtime_dlls(roi_match)
shape_match_copy_runtime_dlls(roi_train)
shape_match_copy_runtime_dlls(shape_match_sample)
shape_match_copy_runtime_dlls(train_smoke)

add_test(NAME train_smoke COMMAND train_smoke)
add_test(NAME roi_train COMMAND roi_train)
add_test(NAME match_smoke COMMAND match_smoke)
add_test(NAME roi_match COMMAND roi_match)

add_test(
    NAME cli_train
    COMMAND shape_match_sample train
        ${CMAKE_CURRENT_SOURCE_DIR}/assets/template.bmp
        ${CMAKE_BINARY_DIR}/cli_artifacts/models/default
)
set_tests_properties(cli_train PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    FIXTURES_SETUP cli_model
)

add_test(
    NAME cli_match
    COMMAND shape_match_sample match
        ${CMAKE_CURRENT_SOURCE_DIR}/assets/scene.bmp
        ${CMAKE_BINARY_DIR}/cli_artifacts/models/default
        ${CMAKE_BINARY_DIR}/cli_artifacts/output/cli_match_overlay.png
)
set_tests_properties(cli_match PROPERTIES
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    FIXTURES_REQUIRED cli_model
)
