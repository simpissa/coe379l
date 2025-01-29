cmake_minimum_required( VERSION 3.13 )
project( jordan VERSION 1.0 )

add_executable( jordan jordan.cxx )
find_package( MPI )
target_include_directories(
        jordan PUBLIC
        ${MPI_C_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} )
target_link_libraries(
        jordan PUBLIC
        ${MPI_C_LIBRARIES} )

install( TARGETS jordan DESTINATION . )
