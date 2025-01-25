#include <iostream>
#include <string>
#include <mpi.h>
using namespace std;

int main(int argc,char **argv) {
        MPI_Init(0,0);
        MPI_Comm comm = MPI_COMM_WORLD;
        int nproc;
        int rank;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &nproc);
	std::string output = std::string("Hello from process ") + std::to_string(rank+1) + std::string(" out of ") + std::to_string(nproc) + std::string("!\n");
	std::cout << output;
  return 0;
}
