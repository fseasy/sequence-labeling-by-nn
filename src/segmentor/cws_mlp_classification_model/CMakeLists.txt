set(exe_name
    cws_mlp_classification
)

add_executable(${exe_name}
               ${exe_name}.cpp
               ${mlp_base_model}
)

target_link_libraries(${exe_name} 
                     ${Boost_LIBRARIES}
                     trivial)
SET_PROPERTY(TARGET ${exe_name} PROPERTY FOLDER "segmentor")