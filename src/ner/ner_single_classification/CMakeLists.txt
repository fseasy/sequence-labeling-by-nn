set(exe_name
    ner_single_classification
)

set(specific_headers
    ${exe_name}_model.h
)
set(specific_libs
    ${exe_name}_model.cpp
)

add_executable(${exe_name}
               ${exe_name}.cpp
               ${single_input_modelhandler_headers} # model handler
               ${specific_headers} # model
               ${single_input_model_headers}        # base model
               ${ner_common_headers}                # common header
               ${specific_libs}
               ${single_input_model_libs} 
               ${ner_common_libs}
               )
               
target_link_libraries(${exe_name}
                      dynet
                      ${Boost_LIBRARIES})
