cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(demucs)

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -g -O0")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -O2 -fdata-sections -ffunction-sections")
endif()

include(cmake/msp_dependencies.cmake)
include(cmake/3rdparty.cmake)
include(cmake/options.cmake)

include_directories(${MSP_INC_DIR})
link_directories(${MSP_LIB_DIR})

include_directories(src)
aux_source_directory(src SRC)
add_executable(${PROJECT_NAME} main.cpp ${SRC})
target_link_libraries(${PROJECT_NAME} ${MSP_LIBS})

add_executable(run_model run_model.cpp ${SRC})
target_link_libraries(run_model ${MSP_LIBS})

install(TARGETS ${PROJECT_NAME} run_model
        RUNTIME
        DESTINATION ./)