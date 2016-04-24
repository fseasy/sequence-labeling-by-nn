INCLUDE_DIRECTORIES(${source_directory})

set(util_directory
    ${source_directory}/utils
)

set(module_directory
    ${source_directory}/modelmodule
)
set(common_headers
    ${util_directory}/typedeclaration.h
    ${util_directory}/utf8processing.hpp
    ${util_directory}/stat.hpp
    ${util_directory}/dict_wrapper.hpp
    ${module_directory}/layers.h
)
set(common_libs
    ${module_directory}/layers.cpp
)
add_subdirectory(postagger)
add_subdirectory(ner)