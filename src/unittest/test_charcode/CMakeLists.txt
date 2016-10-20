ADD_EXECUTABLE(test_charcode
               test_charcode.cpp
               ${charcode_headers}
               ${unittest_framework_include})
               
TARGET_LINK_LIBRARIES(test_charcode
                      trivial
                      ${Boost_LIRARIES})

SET_PROPERTY(TARGET test_charcode PROPERTY FOLDER "unittest")                 
