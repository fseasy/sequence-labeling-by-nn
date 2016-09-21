FILE(GLOB test_charcode_srcs "charcode/*.cpp")         
SOURCE_GROUP("unittest\\test_charcode" FILES ${test_charcode_srcs})
ADD_EXECUTABLE(test_charcode
               test_charcode.cpp
               ${charcode_headers})
               
TARGET_LINK_LIBRARIES(test_charcode
                      trivial)

SET_PROPERTY(TARGET test_charcode PROPERTY FOLDER "unittest")                 