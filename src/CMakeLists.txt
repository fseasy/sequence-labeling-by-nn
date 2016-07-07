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
    ${util_directory}/stash_model.hpp
    ${util_directory}/reader.hpp
    ${util_directory}/general.hpp
    ${module_directory}/layers.h
    ${module_directory}/hyper_layers.h
    ${module_directory}/hyper_input_layers.h
    ${module_directory}/hyper_output_layers.h
)
set(common_libs
    ${module_directory}/layers.cpp
    ${module_directory}/hyper_layers.cpp
    ${module_directory}/hyper_input_layers.cpp
    ${module_directory}/hyper_output_layers.cpp
)

set(additional_base_modules
   # word2vec embedding helper 
   ${util_directory}/word2vec_embedding_helper.h
   ${util_directory}/word2vec_embedding_helper.cpp
)

# context module

set(context_module
    ${module_directory}/context_feature.h
    ${module_directory}/context_feature.cpp
    ${module_directory}/context_feature_extractor.h
    ${module_directory}/context_feature_extractor.cpp
    ${module_directory}/context_feature_layer.h
    ${module_directory}/context_feature_layer.cpp
)

# base reader

set(base_reader_module
    ${util_directory}/reader.hpp
)

add_subdirectory(postagger)
#add_subdirectory(ner)
add_subdirectory(segmentor)