FILE(GLOB layer_all_files "*")

ADD_LIBRARY(layers STATIC ${layer_all_files})

SET_PROPERTY(TARGET layers PROPERTY FOLDER "libraries")