set(segmentor_base_dir
    ${source_directory}/segmentor)
set(base_model_dir
    ${segmentor_base_dir}/base_model
)
set(modelhandler_dir
    ${segmentor_base_dir}/model_handler
)
set(cws_utils_dir
    ${segmentor_base_dir}/cws_utils)

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

# cws utils
set(cws_utils_headers
    ${cws_utils_dir}/cws_utils.h
)
set(cws_utils_libs
    ${cws_utils_dir}/cws_utils.cpp
)

# combine
set(cws_common_headers
    ${common_headers}
    ${cws_utils_headers}
)

set(cws_common_libs
    ${common_libs}
    ${cws_utils_libs}    
)

add_subdirectory(cws_single_classification)
