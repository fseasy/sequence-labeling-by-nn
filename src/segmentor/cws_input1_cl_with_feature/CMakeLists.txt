set(exe_name
    cws_input1_cl_with_feature
)

set(specific_dependencies
    ${exe_name}_model.h
    ${exe_name}_model.cpp
)

add_executable(${exe_name}
               ${exe_name}.cpp
               ${input1_with_feature_modelhandler_0628_dependencies} # model handler
               ${specific_dependencies} # model
               ${input1_with_feature_model_0628_dependencies}        # base model
               ${set_cws_feature_dependencies}
               ${cws_common_headers}                # common header
               ${cws_common_libs}
               )
               
target_link_libraries(${exe_name}
                      cnn
                      ${Boost_LIBRARIES})
