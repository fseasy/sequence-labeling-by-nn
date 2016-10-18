set(exe_name
    pos_input1_mlp_without_tag_no_feature
)

set(specific_headers
    ${exe_name}_model.h
)
set(specific_libs
    ${exe_name}_model.cpp
)

add_executable(${exe_name}
               ${exe_name}.cpp
               ${input1_mlp_no_feature_headers} # base model and handler
               ${specific_headers}   # model
               ${pos_common_headers}                # common header
               ${context_module}
               ${input1_mlp_no_feature_libs}
               ${specific_libs}
               ${pos_common_libs}
               ${pos_reader_module}
               )
               
target_link_libraries(${exe_name}
                      dynet
                      ${Boost_LIBRARIES})
