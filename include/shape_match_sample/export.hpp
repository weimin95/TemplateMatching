#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(SHAPE_MATCH_SAMPLE_BUILD_DLL)
#define SHAPE_MATCH_SAMPLE_API __declspec(dllexport)
#else
#define SHAPE_MATCH_SAMPLE_API __declspec(dllimport)
#endif
#else
#define SHAPE_MATCH_SAMPLE_API
#endif
