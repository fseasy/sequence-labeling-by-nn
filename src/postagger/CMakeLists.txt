set(postagger_base_dir
    ${source_directory}/postagger
)
set(postagger_base_model_dir
    ${postagger_base_dir}/base_model
)
set(postagger_modelhandler_dir
    ${postagger_base_dir}/model_handler
)
set(postagger_modelmodule_dir
    ${postagger_base_dir}/postagger_module
)
# single input model
set(single_input_model_headers
    ${postagger_base_model_dir}/single_input_model.h
)
set(single_input_model_libs
    ${postagger_base_model_dir}/single_input_model.cpp
)

# single input model handler
set(single_input_modelhandler_headers
    ${postagger_modelhandler_dir}/single_input_modelhandler.h
)


# single input with feature 
set(single_input_with_feature_model_hpp
    ${postagger_base_model_dir}/single_input_with_feature_model.hpp
)

# single input with feature modelhandler
set(single_input_with_feature_modelhandler_hpp
    ${postagger_modelhandler_dir}/single_input_with_feature_modelhandler.hpp
)

# model module (features extractor and layer)

set(pos_model_module_hpps
    ${postagger_modelmodule_dir}/pos_feature_extractor.hpp
    ${postagger_modelmodule_dir}/pos_feature_layer.hpp
)

set(pos_common_headers
    ${common_headers}
)

set(pos_common_libs
    ${common_libs}
)

#add_subdirectory(postagger_pretag)
#add_subdirectory(postagger_dc)
#add_subdirectory(postagger_crf)
#add_subdirectory(postagger_crfdc)
add_subdirectory(pos_single_classification)

add_subdirectory(pos_single_classification_with_feature)