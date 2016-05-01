INCLUDE_DIRECTORIES(${source_directory})

set(postagger_headers
    bilstmmodel4tagging.hpp
)

set (doublechannel_postagger_headers
    bilstmmodel4tagging_doublechannel.h
    doublechannel_modelhandler.h
)

set( doublechannel_postagger_libs
    bilstmmodel4tagging_doublechannel.cpp
    doublechannel_modelhandler.cpp
)

set(bilstmcrf_headers
    bilstmcrf.h
    bilstmcrf_modelhandler.h
)
set(bilstmcrf_libs
    bilstmcrf.cpp
    bilstmcrf_modelhandler.cpp
)

ADD_EXECUTABLE(postagger postagger.cpp ${common_headers} ${common_libs} ${postagger_headers})

ADD_EXECUTABLE(postagger_dc postagger_doublechannel.cpp ${common_headers} 
${doublechannel_postagger_headers} ${common_libs} ${doublechannel_postagger_libs})

ADD_EXECUTABLE(postagger_crf postagger_bilstm_crf.cpp ${common_headers} 
${bilstmcrf_headers} ${common_libs} ${bilstmcrf_libs})


target_link_libraries(postagger cnn ${Boost_LIBRARIES})
target_link_libraries(postagger_dc cnn ${Boost_LIBRARIES})
target_link_libraries(postagger_crf cnn ${Boost_LIBRARIES})