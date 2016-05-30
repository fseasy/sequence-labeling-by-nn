set(exe_name
    cws_double_bigram_crf
)
set(base_name
    cws_double_crf
)

set(specific_headers
    ${segmentor_base_dir}/${base_name}/${base_name}_model.h
)
set(specific_libs
    ${segmentor_base_dir}/${base_name}/${base_name}_model.cpp
)

add_executable(${exe_name}
               ${exe_name}.cpp
               ${input2_bigram_modelhandler_headers} # model handler
               ${specific_headers} # model
               ${input2_model_headers}        # base model
               ${cws_common_headers}                # common header
               ${specific_libs}
               ${input2_model_libs} 
               ${cws_common_libs}
               )
               
target_link_libraries(${exe_name}
                      cnn
                      ${Boost_LIBRARIES})
