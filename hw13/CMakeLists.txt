cmake_minimum_required( VERSION 3.12 )
project( matvec VERSION 1.0 )
find_package( MPI )
find_package( PkgConfig REQUIRED )
pkg_check_modules( PETSC REQUIRED petsc )
message( STATUS "PETSc includes: ${PETSC_INCLUDE_DIRS}" )
message( STATUS "PETSc libraries: ${PETSC_LIBRARY_DIRS}" )
add_executable( matvec matvec.c )
target_include_directories(
        matvec PUBLIC
        ${PETSC_INCLUDE_DIRS} )
target_link_directories(
        matvec PUBLIC
        ${PETSC_LIBRARY_DIRS} )
target_link_libraries(
        matvec PUBLIC petsc )
target_include_directories(
        matvec PUBLIC
        ${MPI_C_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} )
target_link_libraries(
        matvec PUBLIC
        ${MPI_C_LIBRARIES} )
install( TARGETS matvec DESTINATION . )
