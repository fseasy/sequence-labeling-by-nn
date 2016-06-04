set(exe_name
    pos_single_classification_with_feature
)

set(specific_headers
    ${exe_name}_model.h
)
set(specific_libs
    ${exe_name}_model.cpp
)

add_executable(${exe_name}
               ${exe_name}.cpp
               ${pos_model_module_hpps}
               ${single_input_with_feature_modelhandler_hpp} # model handler
               ${specific_headers} # model
               ${single_input_with_feature_model_hpp}        # base model
               ${pos_common_headers}                # common header
               ${specific_libs}
               ${pos_common_libs}
               )
               
target_link_libraries(${exe_name}
                      cnn
                      ${Boost_LIBRARIES})
