set(segmentor_base_dir
    ${source_directory}/segmentor)
set(base_model_dir
    ${segmentor_base_dir}/base_model
)
set(modelhandler_dir
    ${segmentor_base_dir}/model_handler
)
set(cws_module_dir
    ${segmentor_base_dir}/cws_module)

# single input model
set(single_input_model_headers
    ${base_model_dir}/single_input_model.h
)
set(single_input_model_libs
    ${base_model_dir}/single_input_model.cpp
)

# single input model handler
set(single_input_modelhandler_headers
    ${modelhandler_dir}/single_input_modelhandler.h
)

# double input model
set(input2_model_headers
    ${base_model_dir}/input2_model.h
)
set(input2_model_libs
    ${base_model_dir}/input2_model.cpp
)
# double input modelhandler
set(input2_modelhandler_headers
    ${modelhandler_dir}/input2_modelhandler.h
)

# cws module
set(cws_module_headers
    ${cws_module_dir}/cws_tagging_system.h
    ${cws_module_dir}/cws_output_layer.h
)
set(cws_module_libs
    ${cws_module_dir}/cws_tagging_system.cpp
    ${cws_module_dir}/cws_output_layer.cpp
)

# combine
set(cws_common_headers
    ${common_headers}
    ${cws_module_headers}
)

set(cws_common_libs
    ${common_libs}
    ${cws_module_libs}    
)

add_subdirectory(cws_single_classification)
add_subdirectory(cws_single_pretag)
add_subdirectory(cws_single_crf)

add_subdirectory(cws_double_classification)
add_subdirectory(unit_test)
