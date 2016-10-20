set(exe_name
     cws_double_classification
)

set(specific_headers
    ${exe_name}_model.h
)
set(specific_libs
    ${exe_name}_model.cpp
)
add_executable(${exe_name}
               ${exe_name}.cpp
               ${input2_modelhandler_headers} # model handler
               ${specific_headers} # model
               ${input2_model_headers}        # base model
               ${cws_common_headers}                # common header
               ${specific_libs}
               ${input2_model_libs} 
               ${cws_common_libs}
               )
               
target_link_libraries(${exe_name}
                      dynet
                      ${Boost_LIBRARIES})
