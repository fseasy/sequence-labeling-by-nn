
include_directories(${source_directory}/trivial)
add_executable(test_charcode
               test_charcode.cpp
               ${charcode_incs}
               )
MESSAGE(STATUS "charcode includes in unittest : " ${charcode_incs})
target_link_libraries(test_charcode
                      charcode)