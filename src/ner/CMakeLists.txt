INCLUDE_DIRECTORIES(${source_directory})

set(ner_base_dir
    ${source_directory}/ner
)

set(ner_base_model_dir
    ${ner_base_dir}/base_model
)

set(ner_modelhandler_dir
    ${ner_base_dir}/model_handler
)

# single input model
set(single_input_model_headers
    ${ner_base_model_dir}/input2D_model.h
)
set(single_input_model_libs
    ${ner_base_model_dir}/input2D_model.cpp
)

# single input model handler
set(single_input_modelhandler_headers
    ${ner_modelhandler_dir}/input2D_modelhandler.h
)

set(ner_common_headers
    ${common_headers}
)

set(ner_common_libs
    ${common_libs}
)

add_subdirectory(ner_cl)   
add_subdirectory(ner_doublechannel)
add_subdirectory(ner_crf_dc)
add_subdirectory(ner_crf)

#add_subdirectory(ner_single_classification)
ADD_SUBDIRECTORY(mlp)