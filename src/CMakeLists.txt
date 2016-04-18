INCLUDE_DIRECTORIES(${source_directory})
ADD_EXECUTABLE(postagger postagger.cc typedeclaration.h utf8processing.hpp 
                                      dict_wrapper.hpp stat.hpp 
                                      bilstmmodel4tagging.hpp)
ADD_EXECUTABLE(ner ner.cc typedeclaration.h utf8processing.hpp 
                                      dict_wrapper.hpp stat.hpp 
                                      bilstmmodel4ner.hpp)                               
target_link_libraries(postagger cnn ${Boost_LIBRARIES})
target_link_libraries(ner cnn ${Boost_LIBRARIES})