cmake_minimum_required( VERSION 3.13 )
project( isendirecvarray VERSION 1.0 )

add_executable( isendirecvarray )
target_sources( isendirecvarray PRIVATE isendirecvarray.cxx )

# find_package( MPI )
# target_include_directories(
# 	isendirecvarray PUBLIC
# 	${MPI_C_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} )
# target_link_libraries(
# 	isendirecvarray PUBLIC 
# 	${MPI_C_LIBRARIES} )

find_package( mpl REQUIRED )
target_include_directories(
        isendirecvarray PUBLIC
        mpl::mpl )
target_link_libraries(
        isendirecvarray PUBLIC
        mpl::mpl )

install( TARGETS isendirecvarray DESTINATION . )
