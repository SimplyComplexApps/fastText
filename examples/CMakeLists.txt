cmake_minimum_required(VERSION 4.0.0)
project(c_api_example)

add_executable(c_api_example c_api_example.c)

target_include_directories(c_api_example PRIVATE ../src)

target_link_libraries(c_api_example PRIVATE fasttext-static)
