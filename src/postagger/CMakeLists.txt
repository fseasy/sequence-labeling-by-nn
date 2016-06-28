include_directories(${source_directory})

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
set(input1_feature2input_layer_model_hpp
    ${single_input_with_feature_model_hpp}
    ${postagger_base_model_dir}/input1_feature2input_layer_model.hpp
)
set(input1_feature2output_layer_model_hpp
    ${single_input_with_feature_model_hpp}
    ${postagger_base_model_dir}/input1_feature2output_layer_model.hpp
)

# bare input1 with feature no merge
set(bareinput1_feature2input_layer_no_merge_model_hpp
    ${single_input_with_feature_model_hpp}
    ${postagger_base_model_dir}/bareinput1_f2i_no_merge_model.hpp)
    
set(bareinput1_feature2output_layer_no_merge_model_hpp
    ${single_input_with_feature_model_hpp}
    ${postagger_base_model_dir}/bareinput1_f2o_no_merge_model.hpp)

set(bareinput1_feature2output_layer_nonlinear_model_hpp
    ${bareinput1_feature2output_layer_no_merge_model_hpp}
    ${postagger_base_model_dir}/bareinput1_f2o_nonlinear_model.hpp)
   
    
# single input with feature modelhandler
set(single_input_with_feature_modelhandler_hpp
    ${postagger_modelhandler_dir}/single_input_with_feature_modelhandler.hpp
)

# input2 with feature
set(input2_with_feature_model_hpp 
${postagger_base_model_dir}/input2_with_feature_model.hpp )

set(input2_feature2input_layer_model_hpp
    ${input2_with_feature_model_hpp}
    ${postagger_base_model_dir}/input2_feature2input_layer_model.hpp)
set(input2_feature2output_layer_model_hpp
    ${input2_with_feature_model_hpp}
    ${postagger_base_model_dir}/input2_feature2output_layer_model.hpp)
# input2 with feature modelhandler
set(input2_with_feature_modelhandler_hpp
    ${postagger_modelhandler_dir}/input2_with_feature_modelhandler.hpp)

# mlp input1 without tag
set(input1_mlp_headers
    ${postagger_base_model_dir}/input1_mlp_model.h
    ${postagger_modelhandler_dir}/input1_mlp_modelhandler.h
)
set(input1_mlp_libs
    ${postagger_base_model_dir}/input1_mlp_model.cpp
    ${postagger_modelhandler_dir}/input1_mlp_modelhandler.cpp
)

# mlp input1 without tag NO FEATURE
set(input1_mlp_no_feature_headers
    ${postagger_base_model_dir}/input1_mlp_model_no_feature.h
    ${postagger_modelhandler_dir}/input1_mlp_modelhandler_no_feature.h
)
set(input1_mlp_no_feature_libs
    ${postagger_base_model_dir}/input1_mlp_model_no_feature.cpp
)


# with feature model module (features extractor and layer)

set(pos_feature_module_headers
    ${postagger_modelmodule_dir}/pos_feature_extractor.h
    ${postagger_modelmodule_dir}/pos_feature_layer.h
    ${postagger_modelmodule_dir}/pos_feature.h
)

set(pos_feature_module_libs
    ${postagger_modelmodule_dir}/pos_feature_extractor.cpp
    ${postagger_modelmodule_dir}/pos_feature_layer.cpp
    ${postagger_modelmodule_dir}/pos_feature.cpp
)

# reader specifics
set(pos_reader_module
    ${postagger_modelmodule_dir}/pos_reader.h
    ${postagger_modelmodule_dir}/pos_reader.cpp
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
#add_subdirectory(pos_single_classification)

#add_subdirectory(pos_single_classification_with_feature)
#add_subdirectory(pos_single_pretag_with_feature)
#add_subdirectory(pos_single_crf_with_feature)

#add_subdirectory(pos_input2_classification_with_feature)
#add_subdirectory(pos_input2_pretag_with_feature)
#add_subdirectory(pos_input2_crf_with_feature)

add_subdirectory(pos_bareinput1_classification_with_feature_no_merge)

#add_subdirectory(pos_input1_mlp_without_tag)
#add_subdirectory(pos_input1_mlp_without_tag_NO_FEATURE)
