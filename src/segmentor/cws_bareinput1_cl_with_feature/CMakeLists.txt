#F2I
set(exe_name
    cws_bareinput1_cl_f2i
)

set(specific_dependencies
    ${exe_name}_model.h
    ${exe_name}_model.cpp
)

add_executable(${exe_name}
               ${exe_name}.cpp
               ${input1_with_feature_modelhandler_0628_dependencies} # model handler
               ${specific_dependencies} # model
               ${bareinput1_f2i_model_dependencies}        # base model
               ${context_module}
               ${set_cws_feature_dependencies}
               ${cws_reader_module}
               ${cws_common_headers}                # common header
               ${cws_common_libs}
               )
               
target_link_libraries(${exe_name}
                      cnn
                      ${Boost_LIBRARIES})


# F2O                
set(exe_name
    cws_bareinput1_cl_f2o
)

set(specific_dependencies
    ${exe_name}_model.h
    ${exe_name}_model.cpp
)

add_executable(${exe_name}
               ${exe_name}.cpp
               ${input1_with_feature_modelhandler_0628_dependencies} # model handler
               ${specific_dependencies} # model
               ${bareinput1_f2o_model_dependencies}        # base model
               ${context_module}
               ${set_cws_feature_dependencies}
               ${cws_reader_module}
               ${cws_common_headers}                # common header
               ${cws_common_libs}
               )
               
target_link_libraries(${exe_name}
                      cnn
                      ${Boost_LIBRARIES})