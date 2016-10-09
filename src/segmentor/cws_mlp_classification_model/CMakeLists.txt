FILE(GLOB dep_token_module "${cws_module_token_dir}/cws_basic_token_module.*"
                           "${cws_module_token_dir}/cws_tag_definition.h"
                           "${cws_module_token_dir}/cws_tag_utility.h")
FILE(GLOB dep_structure_param_module "${cws_module_structure_param_dir}/basic_mlp_param.*")
FILE(GLOB dep_nn_module "${cws_module_nn_dir}/nn_common_interface.*"
                        "${cws_module_nn_dir}/nn_common_interface_cnn_impl.*")
FILE(GLOB dep_modelhandler "${cws_module_dir}/cws_general_modelhandler.*")

set(exe_name
    cws_mlp_classification
)

add_executable(${exe_name}
               ${exe_name}.cpp
               ${mlp_base_model}
               ${dep_token_module}
               ${dep_structure_param_module}
               ${dep_nn_module}
               ${dep_modelhandler}
)

target_link_libraries(${exe_name} 
                      ${Boost_LIBRARIES}
                      trivial)
SET_PROPERTY(TARGET ${exe_name} PROPERTY FOLDER "segmentor")