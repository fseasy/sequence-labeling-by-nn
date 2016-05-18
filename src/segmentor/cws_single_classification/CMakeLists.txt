
set(cws_single_classification_headers
    cws_single_classification_model.hpp
)
add_executable(cws_single_classification 
               cws_single_classification.cpp
               ${cws_single_classification_headers}
               ${cws_utils_headers}
               ${common_headers}
               ${single_model_headers}
               ${single_modelhandler_headers}
               ${common_libs}
               ${single_modelhandler_header})
               
target_link_libraries(cws_single_classification 
                      cnn
                      ${Boost_LIBRARIES})