project(NeoLithicLTP)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -O3 -g")
endif()

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

enable_testing()

## COPY from project DYNET.CMakeLists

### 1. find mkl

function(find_mkl)
  set(MKL_ARCH intel64)
  find_path(MKL_INCLUDE_DIR mkl.h
            PATHS ${MKL_ROOT} ${MKL_ROOT}/include)
  find_library(MKL_CORE_LIB NAMES mkl_intel_lp64 mkl_intel_thread mkl_core
               PATHS ${MKL_ROOT} ${MKL_ROOT}/lib/${MKL_ARCH}
               DOC "MKL core library path")
  find_library(MKL_COMPILER_LIB NAMES libiomp5md
               PATHS ${MKL_ROOT} ${MKL_ROOT}/../compiler/lib/${MKL_ARCH}
               DOC "MKL compiler lib (for threaded MKL)")
  if(MKL_INCLUDE_DIR AND MKL_CORE_LIB AND MKL_COMPILER_LIB)
    get_filename_component(MKL_CORE_LIB_DIR ${MKL_CORE_LIB} DIRECTORY)
    get_filename_component(MKL_COMPILER_LIB_DIR ${MKL_COMPILER_LIB} DIRECTORY)
    message(STATUS "Found MKL\n   * include: ${MKL_INCLUDE_DIR},\n   * core library dir: ${MKL_CORE_LIB_DIR},\n   * compiler library dir: ${MKL_COMPILER_LIB_DIR}")
    set(MKL_LIBS mkl_intel_lp64.lib mkl_intel_thread.lib mkl_core.lib libiomp5md.lib PARENT_SCOPE)
    include_directories(${MKL_INCLUDE_DIR})
    link_directories(${MKL_CORE_LIB_DIR} ${MKL_COMPILER_LIB_DIR})
  else()
    message(FATAL_ERROR "Failed to find MKL in path: ${MKL_ROOT} (Did you set MKL_ROOT properly?)")
  endif()
endfunction()

### 2. find cudnn

function(find_cudnn)
  set(CUDNN_ROOT "" CACHE PATH "CUDNN root path")
  find_path(CUDNN_INCLUDE_DIRS cudnn.h
    PATHS ${CUDNN_ROOT}
          ${CUDNN_ROOT}/include
    DOC "CUDNN include path")
  find_library(CUDNN_LIBRARIES NAMES libcudnn.so
    PATHS ${CUDNN_ROOT}
          ${CUDNN_ROOT}/lib
          ${CUDNN_ROOT}/lib64
    DOC "CUDNN library path")
  if(CUDNN_INCLUDE_DIRS AND CUDNN_LIBRARIES)
    set(CUDNN_FOUND TRUE PARENT_SCOPE)
    message(STATUS "Found CUDNN (include: ${CUDNN_INCLUDE_DIRS}, library: ${CUDNN_LIBRARIES})")
    mark_as_advanced(CUDNN_INCLUDE_DIRS CUDNN_LIBRARIES)
  else()
    MESSAGE(FATAL_ERROR "Failed to find CUDNN in path: ${CUDNN_ROOT} (Did you set CUDNN_ROOT properly?)")
  endif()
endfunction()

### find Boost

if(DEFINED ENV{BOOST_ROOT})
  set(Boost_NO_SYSTEM_PATHS ON)
  get_filename_component(Boost_INCLUDE_DIR "${Boost_INCLUDE_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
endif()
set(Boost_REALPATH ON)
message("-- Boost dir is " ${Boost_INCLUDE_DIR})
if(WIN32)
    set(Boost_USE_STATIC_LIBS ON)
    find_package(Boost COMPONENTS program_options serialization log_setup log thread system filesystem date_time chrono REQUIRED)
else()
    add_definitions (-DBOOST_LOG_DYN_LINK)
    find_package(Boost COMPONENTS program_options serialization log_setup log thread system REQUIRED)
endif()
include_directories(${Boost_INCLUDE_DIR})
if(WIN32)
    #set(LIBS ${LIBS} ${Boost_LIBRARIES})
    #add_definitions(-DBOOST_ALL_DYN_LINK)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LIBPATH:${Boost_LIBRARY_DIRS}")
    MESSAGE(${LIBS} ${Boost_LIBRARIES})
else()
    set(LIBS ${LIBS} ${Boost_LIBRARIES} pthread)
endif()

### decide backend
 
if(BACKEND)
  message("-- BACKEND: ${BACKEND}")
else()
  message("-- BACKEND not specified, defaulting to eigen.")
  set(BACKEND "eigen")
endif()

if(BACKEND MATCHES "^eigen$")
  set(WITH_EIGEN_BACKEND 1)
elseif(BACKEND MATCHES "^cuda$")
  set(WITH_CUDA_BACKEND 1)
else()
  message(SEND_ERROR "BACKEND must be eigen or cuda")
endif()

if (WITH_CUDA_BACKEND)
  find_package(CUDA REQUIRED)
  set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_ROOT})
  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  #list(APPEND CUDA_LIBRARIES /usr/lib64/libpthread.so)
  MESSAGE("CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
  list(REMOVE_ITEM CUDA_LIBRARIES -lpthread)
  set(LIBS ${LIBS} ${CUDA_LIBRARIES})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_HAS_CUDA_FP16 -DEIGEN_USE_GPU")
  #find_cudnn()
  #include_directories(SYSTEM ${CUDNN_INCLUDE_DIRS})
endif()

### look for Eigen
find_package(Eigen3 REQUIRED)
get_filename_component(EIGEN3_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}" REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
message("-- Eigen dir is " ${EIGEN3_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

FIND_PACKAGE(Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

### compiler options(general)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_FAST_MATH")
if (MKL OR MKL_ROOT)
  find_mkl()  # sets include/lib directories and sets ${MKL_LIBS} needed for linking
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_USE_MKL_ALL")
endif()

### compiler options(specific)

if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNOMINMAX")   # Disable min/max macros in windef.h
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W1 -DEIGEN_HAS_C99_MATH /MP")   # -Wall produces 20k warnings
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -funroll-loops -Wall -std=c++11 -Ofast -g -march=native")
endif()


# dispaly folder in MSVS
SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON) 


SET(thirdparty_dir ${PROJECT_SOURCE_DIR}/3rdparty)
INCLUDE_DIRECTORIES(${thirdparty_dir}/dynet) # for #include "dynet/xxxx"

# build thirdparty libraries(DyNet)
ADD_SUBDIRECTORY(${thirdparty_dir}/dynet/dynet)

SET_PROPERTY(TARGET dynet PROPERTY FOLDER "3rdparty_libraries")
if(WITH_CUDA_BACKEND)
SET_PROPERTY(TARGET dynetcuda PROPERTY FOLDER "3rdparty_libraries")
endif()

SET(source_directory ${PROJECT_SOURCE_DIR}/src) # for compatibility 

SET(source_dir ${PROJECT_SOURCE_DIR}/src) # for new structure
# source
ADD_SUBDIRECTORY(src)
