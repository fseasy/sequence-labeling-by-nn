set(exe_name
    pos_input1_mlp_with_tag
)

set(specific_headers
    ${exe_name}_model.h
)
set(specific_libs
    ${exe_name}_model.cpp
)

add_executable(${exe_name}
               ${exe_name}.cpp
               ${pos_feature_module_headers} # pos model module addition
               ${input1_mlp_headers} # base model and handler
               ${specific_headers}   # model
               ${pos_common_headers}                # common header
               ${context_module}
               ${pos_feature_module_libs}
               ${input1_mlp_libs}
               ${specific_libs}
               ${pos_common_libs}
               ${pos_reader_module}
               )
               
target_link_libraries(${exe_name}
                      dynet
                      ${Boost_LIBRARIES})
