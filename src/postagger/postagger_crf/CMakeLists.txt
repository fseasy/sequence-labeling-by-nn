INCLUDE_DIRECTORIES(${source_directory})

# crf
set(bilstmcrf_headers
    bilstmcrf.h
    bilstmcrf_modelhandler.h
)
set(bilstmcrf_libs
    bilstmcrf.cpp
    bilstmcrf_modelhandler.cpp
)

ADD_EXECUTABLE(postagger_crf postagger_bilstmcrf.cpp ${common_headers} 
${bilstmcrf_headers} ${common_libs} ${bilstmcrf_libs})

target_link_libraries(postagger_crf cnn ${Boost_LIBRARIES})
