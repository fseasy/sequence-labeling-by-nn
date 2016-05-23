set(specific_headers
    cws_single_pretag_model.h
)
set(specific_libs
    cws_single_pretag_model.cpp
)

set(exe_name
     cws_single_pretag
)

add_executable(${exe_name}
               ${exe_name}.cpp
               ${single_input_modelhandler_headers} # model handler
               ${specific_headers} # model
               ${single_input_model_headers}        # base model
               ${cws_common_headers}                # common header
               ${specific_libs}
               ${single_input_model_libs} 
               ${cws_common_libs}
               )
               
target_link_libraries(${exe_name}
                      cnn
                      ${Boost_LIBRARIES})
