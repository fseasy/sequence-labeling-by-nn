INCLUDE_DIRECTORIES(${source_directory})

#  dc
set (doublechannel_postagger_headers
    bilstmmodel4tagging_doublechannel.h
    doublechannel_modelhandler.h
)

set (doublechannel_postagger_libs
    bilstmmodel4tagging_doublechannel.cpp
    doublechannel_modelhandler.cpp    
)

ADD_EXECUTABLE(postagger_dc postagger_doublechannel.cpp ${common_headers} 
${doublechannel_postagger_headers} ${common_libs} ${doublechannel_postagger_libs})

target_link_libraries(postagger_dc cnn ${Boost_LIBRARIES})
