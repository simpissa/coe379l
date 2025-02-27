cmake_minimum_required( VERSION 3.13 )
project( transpose VERSION 1.0 )

add_executable( transpose transpose.cxx )
find_package( MPI )
target_include_directories(
        transpose PUBLIC
        ${MPI_C_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} )
target_link_libraries(
        transpose PUBLIC
        ${MPI_C_LIBRARIES} )

install( TARGETS transpose DESTINATION . )
