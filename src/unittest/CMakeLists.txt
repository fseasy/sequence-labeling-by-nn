ADD_SUBDIRECTORY(test_charcode)
ADD_SUBDIRECTORY(test_lookup_table)
ADD_SUBDIRECTORY(test_cwstag)

FILE(GLOB test_lookup_table_srcs "test_lookup_table/*.cpp")   
FILE(GLOB test_charcode_srcs "test_charcode/*.cpp")                  
FILE(GLOB test_cwstag_srcs "test_cwstag/*.cpp")


SOURCE_GROUP("unittest\\test_lookup_table" FILES ${test_lookup_table_srcs})

SOURCE_GROUP("unittest\\test_charcode" FILES ${test_charcode_srcs})

SOURCE_GROUP("unittest\\test_charcode" FILES ${test_cwstag_srcs})