
add_executable(cws_evaluation_test 
               cws_evaluation.cpp
               ${cws_common_headers} 
               ${cws_common_libs}
               )
target_link_libraries(cws_evaluation_test 
                      cnn
                      ${Boost_LIBRARIES})