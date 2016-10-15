ADD_EXECUTABLE(test_lookup_table
               test_lookup_table.cpp 
               ${unittest_framework_include}
               ${lookup_table_headers}
               )

TARGET_LINK_LIBRARIES(test_lookup_table 
                      trivial
                      ${Boost_LIBRARIES})

SET_PROPERTY(TARGET test_lookup_table PROPERTY FOLDER "unittest")                          
