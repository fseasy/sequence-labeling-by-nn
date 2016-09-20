FILE(GLOB charcode_incs "*.h" "*.hpp")
FILE(GLOB charcode_srcs "*.cpp")

MESSAGE(STATUS "charcode includes in charcode : \n" ${charcode_incs})
MESSAGE(STATUS "charcode sources in charcode : \n" ${charcode_srcs})

add_library(charcode STATIC ${charcode_srcs})