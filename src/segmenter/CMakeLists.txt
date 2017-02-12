set(segmenter_base_dir
    ${source_directory}/segmenter)
set(base_model_dir
    ${segmenter_base_dir}/base_model
)
set(modelhandler_dir
    ${segmenter_base_dir}/model_handler
)
set(cws_module_dir
    ${segmenter_base_dir}/cws_module)

# single input model
set(single_input_model_headers
    ${base_model_dir}/single_input_model.h
)
set(single_input_model_libs
    ${base_model_dir}/single_input_model.cpp
)

# single input model handler
set(single_input_modelhandler_headers
    ${modelhandler_dir}/single_input_modelhandler.h
)

# single input bigram heandler
set(single_input_bigram_modelhandler_headers
    ${modelhandler_dir}/single_input_bigram_modelhandler.h
    ${single_input_modelhandler_headers}
)
## input model with feature (2016-06-28 )
# input modelhandler 0628
set(input1_with_feature_model_0628_dependencies
    ${base_model_dir}/input1_with_feature_model_0628.hpp
)

set(input1_f2i_model_0628_dependencies
    ${input1_with_feature_model_0628_dependencies}
    ${base_model_dir}/input1_f2i_model_0628.hpp
)

set(input1_f2o_model_0628_dependencies
    ${input1_with_feature_model_0628_dependencies}
    ${base_model_dir}/input1_f2o_model_0628.hpp
)

set(bareinput1_f2i_model_dependencies
    ${input1_with_feature_model_0628_dependencies}
    ${base_model_dir}/bareinput1_f2i_model.hpp
)

set(bareinput1_f2o_model_dependencies
    ${input1_with_feature_model_0628_dependencies}
    ${base_model_dir}/bareinput1_f2o_model.hpp
)

set(input1_with_feature_modelhandler_0628_dependencies
    ${modelhandler_dir}/input1_with_feature_modelhandler_0628.hpp
)

# double input model
set(input2_model_headers
    ${base_model_dir}/input2_model.h
)
set(input2_model_libs
    ${base_model_dir}/input2_model.cpp
)
# double input modelhandler
set(input2_modelhandler_headers
    ${modelhandler_dir}/input2_modelhandler.h
)

# double input bigram modelhandler
set(input2_bigram_modelhandler_headers
    ${modelhandler_dir}/input2_bigram_modelhandler.h
    ${input2_modelhandler_headers}
)

## CWS Feature

set(set_cws_feature_dependencies
    ${cws_module_dir}/cws_feature.h
    ${cws_module_dir}/cws_feature.cpp
    ${cws_module_dir}/cws_feature_layer.h
    ${cws_module_dir}/cws_feature_layer.cpp
    ${cws_module_dir}/lexicon_feature.h
    ${cws_module_dir}/lexicon_feature.cpp
    ${cws_module_dir}/lexicon_feature_layer.h
    ${cws_module_dir}/lexicon_feature_layer.cpp
    ${cws_module_dir}/type_feature.h
    ${cws_module_dir}/type_feature.cpp
)

# CWS Reader

set(cws_reader_module
    ${base_reader_module}
    ${cws_module_dir}/cws_reader.h
    ${cws_module_dir}/cws_reader.cpp
)

# cws module
set(cws_module_headers
    ${cws_module_dir}/cws_tagging_system.h
    ${cws_module_dir}/cws_output_layer.h
    ${cws_module_dir}/nn_module/experiment_layer/nn_cws_specific_output_layer.h
)
set(cws_module_libs
    ${cws_module_dir}/cws_tagging_system.cpp
    ${cws_module_dir}/cws_output_layer.cpp
    ${cws_module_dir}/nn_module/experiment_layer/nn_cws_specific_output_layer.cpp
)


# combine
set(cws_common_headers
    ${common_headers}
    ${cws_module_headers}
)

set(cws_common_libs
    ${common_libs}
    ${cws_module_libs}    
)

#########################  New Structure  ######################

#################
# Segmentor
#################
set(cws_module_token_dir "${cws_module_dir}/token_module")
set(cws_module_structure_param_dir "${cws_module_dir}/structure_param_module")
set(cws_module_nn_dir "${cws_module_dir}/nn_module/")

FILE(GLOB segmenter_token_module_files "${cws_module_token_dir}/*")
FILE(GLOB segmenter_structure_param_module_files "${cws_module_structure_param_dir}/*")
FILE(GLOB segmenter_nn_module_files "${cws_module_nn_dir}/*")

SOURCE_GROUP("cws_module\\token_module" FILES ${segmenter_token_module_files})
SOURCE_GROUP("cws_module\\structure_param_module" FILES ${segmenter_structure_param_module_files})
SOURCE_GROUP("cws_module\\nn_module" FILES ${segmenter_nn_module_files})

#add_subdirectory(cws_single_classification)

#add_subdirectory(cws_single_pretag)
#add_subdirectory(cws_single_crf)

#add_subdirectory(cws_double_classification)
#add_subdirectory(cws_double_pretag)
#add_subdirectory(cws_double_crf)

#add_subdirectory(cws_single_bigram_classification)
#add_subdirectory(cws_single_bigram_pretag)
#add_subdirectory(cws_single_bigram_crf)

#add_subdirectory(cws_double_bigram_classification)
#add_subdirectory(cws_double_bigram_pretag)
#add_subdirectory(cws_double_bigram_crf)

#add_subdirectory(cws_input1_cl_with_feature)
ADD_SUBDIRECTORY(cws_bareinput1_cl_with_feature)
ADD_SUBDIRECTORY(cws_mlp_input1)
ADD_SUBDIRECTORY(cws_rnn_input1)
#add_subdirectory(unit_test)
