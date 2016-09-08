set(exe_name
    cws_mlp_classification
)

add_executable(${exe_name}
               ${exe_name}.cpp
               ${mlp_base_model}
               ${cws_task_specifics}
)

target_link_libraries(${exe_name} 
                     ${Boost_LIBRARIES})