cmake_minimum_required( VERSION 3.13 )
project( hw1 VERSION 1.0 )

add_executable( hw1 hw1.cxx )
find_package( MPI )
target_include_directories(
        hw1 PUBLIC
        ${MPI_C_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} )
target_link_libraries(
        hw1 PUBLIC
        ${MPI_C_LIBRARIES} )

install( TARGETS hw1 DESTINATION . )
