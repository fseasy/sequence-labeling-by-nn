FILE(GLOB dep_token_module "${cws_module_token_dir}/cws_basic_token_module.*"
                           "${cws_module_token_dir}/cws_tag_definition.h"
                           "${cws_module_token_dir}/cws_tag_utility.h")
FILE(GLOB dep_structure_param_module "${cws_module_structure_param_dir}/basic_mlp_param.*")
FILE(GLOB dep_nn_general_module "${cws_module_dir}/cws_output_layer.*"                          # segmenter output layer.
                                "${cws_module_nn_dir}/nn_common_interface.*"
                                "${cws_module_nn_dir}/nn_common_interface_dynet_impl.*"
                                "${cws_module_nn_dir}/mlp_input1/nn_cws_mlp_input1_abstract.*")
FILE(GLOB dep_nn_specific_module "${cws_module_nn_dir}/mlp_input1/nn_cws_mlp_input1_cl.*")            
FILE(GLOB dep_modelhandler "${cws_module_dir}/cws_general_modelhandler.*")
FILE(GLOB dep_eval "${cws_module_dir}/cws_eval.*")
FILE(GLOB dep_stat "${cws_module_dir}/cws_stat.*")
FILE(GLOB dep_io "${cws_module_dir}/cws_reader_unicode.*" "${cws_module_dir}/cws_writer.*")
FILE(GLOB dep_others "${utils_dir}/nn_utility.*")

FILE(GLOB dep_nn_4compatibility "${cws_module_dir}/cws_tagging_system.*")

FILE(GLOB model_template "cws_mlp_input1_template.*")

FILE(GLOB cl_model_files "classification/cws_mlp_input1_cl_model.*")

set(exe_name
    cws_mlp_input1_classification
)

ADD_LIBRARY(cl_model
           STATIC
           ${dep_token_module}
           ${dep_structure_param_module}
           ${dep_nn_general_module}
           ${dep_nn_specific_module}
           ${dep_eval}
           ${dep_stat}
           ${dep_io}
           ${dep_others}
           ${dep_nn_4compatibility}
           ${model_template}
           ${cl_model_files}
)
# because cl_model for cws layers depend on the general layers. What's more, exe needs both cl_mode and layers. IS IT OK in logic?
# ( I think it is ok. )
TARGET_LINK_LIBRARIES(cl_model
                     layers) 

add_executable(${exe_name}
               "classification/${exe_name}.cpp"
               ${dep_modelhandler}
)


if (WITH_CUDA_BACKEND)
    target_link_libraries(${exe_name} gdynet ${Boost_LIBRARIES} trivial layers cl_model)
    add_dependencies(${exe_name} dynetcuda)
    target_link_libraries(${exe_name} dynetcuda)
    CUDA_ADD_CUBLAS_TO_TARGET(${exe_name})
  else()
    target_link_libraries(${exe_name} dynet ${Boost_LIBRARIES} trivial layers cl_model)
endif (WITH_CUDA_BACKEND)

                      
SET_PROPERTY(TARGET cl_model PROPERTY FOLDER "segmenter/mlp_input1/lib")
SET_PROPERTY(TARGET ${exe_name}  PROPERTY FOLDER "segmenter/mlp_input1")
