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

set(single_model_headers
    ${base_model_dir}/single_input_model.h
)
set(single_modelhandler_headers
    ${modelhandler_dir}/single_input_modelhandler.hpp
)
set(cws_utils_headers
    ${cws_utils_dir}/cws_utils.hpp)

add_subdirectory(cws_single_classification)