ADD_SUBDIRECTORY(test_charcode)
ADD_SUBDIRECTORY(test_lookup_table)


FILE(GLOB test_lookup_table_srcs "test_lookup_table/*.cpp")   
FILE(GLOB test_charcode_srcs "charcode/*.cpp")                  

SOURCE_GROUP("unittest\\test_lookup_table" FILES ${test_lookup_table_srcs})

SOURCE_GROUP("unittest\\test_charcode" FILES ${test_charcode_srcs})
