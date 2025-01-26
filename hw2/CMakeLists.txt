cmake_minimum_required( VERSION 3.13 )
project( randommax VERSION 1.0 )

add_executable( randommax randommax.cxx )
find_package( MPI )
target_include_directories(
        randommax PUBLIC
        ${MPI_C_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} )
target_link_libraries(
        randommax PUBLIC
        ${MPI_C_LIBRARIES} )

install( TARGETS randommax DESTINATION . )
