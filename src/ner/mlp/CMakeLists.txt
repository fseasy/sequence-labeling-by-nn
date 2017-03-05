FILE(GLOB dep_token_module ${ner_module}/token_module.*)
FILE(GLOB dep_structure_param_module ${ner_module}/structure_param_module*)
FILE(GLOB dep_nn_module ${ner_module}/nn_module*)
FILE(GLOB model_utility ${ner_module}/model_utility*)
FILE(GLOB model_handler ${ner_module}/model_handler*)

FILE(GLOB dep_io ${ner_module}/ner_reader.* 
				 ${source_directory}/utils/reader.* 
				 ${source_directory}/utils/writer.*)

SET(program_name "cl")

ADD_EXECUTABLE( ${program_name}
				${program_name}.cpp
				${dep_token_module}
				${dep_structure_param_module}
				${dep_nn_module}
				${model_utility}
				${model_handler}
				${dep_io}
				)

if (WITH_CUDA_BACKEND)
	target_link_libraries(${program_name} gdynet ${Boost_LIBRARIES} trivial layers)
	add_dependencies(${program_name} dynetcuda)
	target_link_libraries(${program_name} dynetcuda)
	CUDA_ADD_CUBLAS_TO_TARGET(${program_name})
  else()
	target_link_libraries(${program_name} dynet ${Boost_LIBRARIES} trivial layers)
endif (WITH_CUDA_BACKEND)

SET_PROPERTY(TARGET ${program_name} PROPERTY FOLDER "ner/mlp")