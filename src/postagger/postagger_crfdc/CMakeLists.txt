INCLUDE_DIRECTORIES(${source_directory})

# crf dc
set(bilstmcrf_dc_headers
    bilstmcrf_dc.h
    bilstmcrf_dc_modelhandler.h
)
set(bilstmcrf_dc_libs
    bilstmcrf_dc.cpp
    bilstmcrf_dc_modelhandler.cpp
)

ADD_EXECUTABLE(postagger_crf_dc postagger_bilstmcrf_dc.cpp ${common_headers} 
${bilstmcrf_dc_headers} ${common_libs} ${bilstmcrf_dc_libs})

target_link_libraries(postagger_crf_dc dynet ${Boost_LIBRARIES})
