FILE(GLOB dep_token_module token_module.*)
FILE(GLOB dep_structure_param_module structure_param_module*)
FILE(GLOB dep_nn_module nn_module4cl*)

SET(program_name "cl")

ADD_EXECUTABLE( ${program_name}
				${program_name}.cpp
				${dep_token_module}
				${dep_structure_param_module}
				${dep_nn_module}
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