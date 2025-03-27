cmake_minimum_required( VERSION 3.20 )
project( sudoku VERSION 1.0 )

find_package(OpenMP)
if(OpenMP_C_FOUND)
else()
	message( FATAL_ERROR "Could not find OpenMP" )
endif()

add_executable( sudoku sudoku.c )
target_compile_features( sudoku PRIVATE c_std_99 )
target_link_libraries( sudoku PUBLIC OpenMP::OpenMP_C )

# some compilers (Intel) need this explicitly:
target_link_libraries( sudoku PRIVATE m)

install( TARGETS sudoku DESTINATION . )
