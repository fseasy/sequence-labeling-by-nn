FILE(GLOB lookup_table_headers "lookup_table/*.h*")
FILE(GLOB lookup_table_srcs "lookup_table/*.cpp")

FILE(GLOB charcode_headers "charcode/*.h*")
FILE(GLOB charcode_srcs "charcode/*.cpp")

SET(trivial_headers ${lookup_table_headers}
                    ${charcode_headers})
SET(trivial_srcs ${lookup_table_srcs}
                 ${charcode_srcs})                    

             
ADD_LIBRARY(trivial STATIC ${trivial_headers}
                           ${trivial_srcs})


# GROUP files (For MSVS)
SOURCE_GROUP("lookup_table" FILES ${lookup_table_headers}
                                  ${lookup_table_srcs})

SOURCE_GROUP("charcode" FILES ${charcode_headers}
                              ${charcode_srcs}) 

SET_PROPERTY(TARGET trivial PROPERTY FOLDER "libiraries")                                