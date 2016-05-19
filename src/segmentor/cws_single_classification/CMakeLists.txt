set(cws_single_classification_headers
    cws_single_classification_model.h
)
set(cws_single_classification_libs
    cws_single_classification_model.cpp
)

add_executable(cws_single_classification 
               cws_single_classification.cpp
               ${single_input_modelhandler_headers} # model handler
               ${cws_single_classification_headers} # model
               ${single_input_model_headers}        # base model
               ${cws_common_headers}                # common header
               ${cws_single_classification_libs}
               ${single_input_model_libs} 
               ${cws_common_libs}
               )
               
target_link_libraries(cws_single_classification 
                      cnn
                      ${Boost_LIBRARIES})
