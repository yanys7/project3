# Check cmake version.
# ---------------------------
cmake_minimum_required (VERSION 2.8)

INCLUDE(GNUInstallDirs)

# Configure the path structure.
# ---------------------------
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_RELEASE_POSTFIX "")
set (CMAKE_DEBUG_POSTFIX "-debug")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

# Project specification.
# ---------------------------
#project (PROJECT)
project ($ENV{PROJECT_NAME})

# Custom CFLAGS
# ---------------------------
find_program (ICC_PATH "icc")
find_program (ICPC_PATH "icpc")
set (CMAKE_C_COMPILER ${ICC_PATH})
set (CMAKE_CXX_COMPILER ${ICPC_PATH})
set (PROJECT_CFLAGS "-std=c++11 -O3 -qopenmp -DUSE_BLAS -DUSE_R")
set (MKLROOT "$ENV{HOME}/intel/mkl")
set (PROJECT_EXE_LINKER_FLAGS "-mkl=parallel")

# Setup all flags
# ---------------------------
set (CMAKE_C_FLAGS        "${CMAKE_C_FLAGS} ${PROJECT_CFLAGS}")
set (CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS} ${PROJECT_CFLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${PROJECT_EXE_LINKER_FLAGS}")

# GSL dependency
# ---------------------------
FIND_PACKAGE(GSL REQUIRED)
MESSAGE(STATUS ${GSL_INCLUDE_DIRS})
MESSAGE(STATUS ${GSL_LIBRARIES})

# R dependency
# ---------------------------
FIND_PACKAGE(R REQUIRED)
MESSAGE(STATUS ${R_ROOT_DIR})
MESSAGE(STATUS ${R_INCLUDE_DIRS})
MESSAGE(STATUS ${R_LIBRARIES})

## Rcpp dependency
## ---------------------------
#SET(RCPP_ROOT_DIR "/net/wonderland/home/yanys/R/x86_64-pc-linux-gnu-library/2.13/Rcpp")
#SET(RCPP_INCLUDE_DIRS "${RCPP_ROOT_DIR}/include")
#FIND_LIBRARY(RCPP_LIBRARIES Rcpp PATHS "${RCPP_ROOT_DIR}/lib")
#IF(RCPP_LIBRARIES-NOTFOUND)
#  MESSAGE(STATUS "Rcpp-NOTFOUND")
#ELSE()
#  MESSAGE(STATUS ${RCPP_LIBRARIES})
#ENDIF()



# Headers.
# ---------------------------
include_directories("include" "frame" "external") 
include_directories("external/BayesLogit/src") 
include_directories(${R_INCLUDE_DIRS}) 
include_directories(${RCPP_INCLUDE_DIRS}) 
include_directories("$ENV{HOME}/Bayesian_mediation/RealSimulation/BF_simulation/Bayesian-Mediation-Analysis/gsl/include/gsl") 
include_directories("$ENV{HOME}/Mixture_mediation/Gaussian-Mixture-Mediation/boost_1_61_0/boost/math/distributions")
file (GLOB PROJECT_HDRS ${CMAKE_SOURCE_DIR}/include/*)
install (FILES ${PROJECT_HDRS} DESTINATION ${CMAKE_BINARY_DIR}/include)


# Library dependencies
# ---------------------------
file (GLOB CXX_SRC ${CMAKE_SOURCE_DIR}/src/*.cpp)
file (GLOB EXT_CXX_SRC ${CMAKE_SOURCE_DIR}/external/*.cpp)
file (GLOB BAYES_LOGIT_CXX_SRC ${CMAKE_SOURCE_DIR}/external/BayesLogit/src/*.cpp)
add_library ($ENV{PROJECT_NAME} STATIC ${CXX_SRC} ${EXT_CXX_SRC} ${BAYES_LOGIT_CXX_SRC})

# Build the executable files.
# ---------------------------
file (GLOB TEST_CXX_SRC ${CMAKE_SOURCE_DIR}/test/*.cpp)
foreach( testsrc ${TEST_CXX_SRC} )
  string( REPLACE ".cpp" ".x" testexe ${testsrc} )
  get_filename_component( testexe ${testsrc} NAME_WE )
  add_executable( "${testexe}.x" ${testsrc} )
  target_link_libraries( "${testexe}.x" $ENV{PROJECT_NAME} )
  target_link_libraries( "${testexe}.x" ${GSL_LIBRARIES} )
  target_link_libraries( "${testexe}.x" ${R_LIBRARIES} )
  target_link_libraries( "${testexe}.x" ${RCPP_LIBRARIES} )
endforeach()
