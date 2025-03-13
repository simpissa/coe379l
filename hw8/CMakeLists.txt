cmake_minimum_required( VERSION 3.20 )
project( reductcoordinate VERSION 1.0 )

find_package(OpenMP)
if(OpenMP_C_FOUND)
else()
	message( FATAL_ERROR "Could not find OpenMP" )
endif()

add_executable( reductcoordinate reductcoordinate.c )
target_compile_features( reductcoordinate PRIVATE c_std_99 )
target_link_libraries( reductcoordinate PUBLIC OpenMP::OpenMP_C )

# some compilers (Intel) need this explicitly:
target_link_libraries( reductcoordinate PRIVATE m)

install( TARGETS reductcoordinate DESTINATION . )
