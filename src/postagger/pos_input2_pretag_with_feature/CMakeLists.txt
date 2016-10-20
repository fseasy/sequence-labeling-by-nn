set(feature2input_exe_name
    pos_input2_pretag_feature2input_layer
)

set(specific_headers
    ${feature2input_exe_name}_model.h
)
set(specific_libs
    ${feature2input_exe_name}_model.cpp
)

add_executable(${feature2input_exe_name}
               ${feature2input_exe_name}.cpp
               ${pos_feature_module_headers}
               ${input2_with_feature_modelhandler_hpp} # model handler
               ${specific_headers} # model
               ${input2_feature2input_layer_model_hpp}        # feature to input base model 
               ${pos_common_headers}                # common header
               ${specific_libs}
               ${pos_common_libs}
               ${pos_feature_module_libs}
               ${pos_reader_module}
               ${additional_base_modules}
               )
               
target_link_libraries(${feature2input_exe_name}
                      dynet
                      ${Boost_LIBRARIES})

############# feature2output  ###########   
                      
set(feature2output_exe_name
    pos_input2_pretag_feature2output_layer
)

set(specific_headers
    ${feature2output_exe_name}_model.h
)
set(specific_libs
    ${feature2output_exe_name}_model.cpp
)

add_executable(${feature2output_exe_name}
               ${feature2output_exe_name}.cpp
               ${pos_feature_module_headers}
               ${input2_with_feature_modelhandler_hpp} # model handler
               ${specific_headers} # model
               ${input2_feature2output_layer_model_hpp}        # feature to output base model 
               ${pos_common_headers}                # common header
               ${specific_libs}
               ${pos_common_libs}
               ${pos_feature_module_libs}
               ${pos_reader_module}
               ${additional_base_modules}
               )
               
target_link_libraries(${feature2output_exe_name}
                      dynet
                      ${Boost_LIBRARIES})