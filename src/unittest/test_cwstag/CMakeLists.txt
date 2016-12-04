
SET(cws_module_dir "${source_dir}/segmenter/cws_module")

FILE(GLOB dep_files 
          ${cws_module_dir}/cws_eval.*
          ${cws_module_dir}/token_module/cws_tag_definition.*)

ADD_EXECUTABLE( test_cwstag
                test_cwstag.cpp
                ${dep_files}
                ${unittest_framework_include})
                
SET_PROPERTY(TARGET test_cwstag PROPERTY FOLDER "unittest")