/****************************************************************
 ****
 **** This program file is part of the book 
 **** `Parallel programming for Science and Engineering'
 **** by Victor Eijkhout, eijkhout@tacc.utexas.edu
 ****
 **** copyright Victor Eijkhout 2021-2023
 ****
 **** MPI Exercise for Isend/Irecv using MPL
 ****
 ****************************************************************/

#include <iostream>
#include <sstream>
using namespace std;
#include <mpl/mpl.hpp>

#include <cassert>

bool isapprox(int x,int y) {
  return x==y;
};
bool isapprox(double x,double y) {
 return
   ( ( x==0. && fabs(y)<1.e-14 ) || ( y==0. && fabs(x)<1.e-14 ) ||      \
     fabs(x-y)/fabs(x)<1.e-14 );
};
bool isapprox(float x,float y) {
 return
   ( ( x==0. && fabs(y)<1.e-14 ) || ( y==0. && fabs(x)<1.e-14 ) ||      \
     fabs(x-y)/fabs(x)<1.e-14 );
};

double array_error(const vector<double>& ref_array,const vector<double>& value_array) {
  int array_size = ref_array.size();
  assert(array_size==value_array.size());
  double error = 0.,max_value=-1,min_value=-1;
  for (int i=0; i<array_size; i++) {
    double ref = ref_array[i], val = fabs(value_array[i]);
    if (min_value<0 || val<min_value) min_value = val;
    if (max_value<0 || val>max_value) max_value = val;
    double
      rel = (ref - val)/ref,
      e = fabs(rel);
    if (e>error) error = e;
  }
  return error;
};

int test_all_the_same_int( int value,const mpl::communicator& comm ) {
  int final_min,final_max;
  comm.allreduce(mpl::min<int>(),value,final_min);
  //  MPI_Allreduce(&value,&final_min,1,MPI_INT,MPI_MIN,comm);
  comm.allreduce(mpl::max<int>(),value,final_max);
  // MPI_Allreduce(&value,&final_max,1,MPI_INT,MPI_MAX,comm);
  return final_min==final_max;
}
void error_process_print(int error_proc, const mpl::communicator& comm) {
  int nprocs,procno;
  nprocs = comm.size();
  procno = comm.rank();
  if (procno==0) {
    if (error_proc==nprocs)
      printf("Finished; all results correct\n");
    else
      printf("First error occurred on proc %d\n",error_proc);
  }
}
void print_final_result( bool cond,const mpl::communicator& comm ) {
  int nprocs,procno;
  nprocs = comm.size();
  procno = comm.rank();
  int error=nprocs, error_proc=-1;
  if (cond)
    error = procno;
  comm.allreduce(mpl::min<int>(),error,error_proc);
  error_process_print(error_proc,comm);
};


int main() {

  const mpl::communicator &comm_world = mpl::environment::comm_world();
  MPI_Comm_set_errhandler(MPI_COMM_WORLD,MPI_ERRORS_RETURN);
  int nprocs,procno;
  // compute communicator rank and size
  nprocs = comm_world.size();
  procno = comm_world.rank();

  stringstream proctext;

#define N 100
  vector<double> indata(N,1.), outdata(N);
  double leftdata=0.,rightdata=0.;
  int sendto,recvfrom;
  mpl::irequest_pool requests;

  // Exercise:
  // -- set `sendto' and `recvfrom' twice
  //    once to get data from the left, once from the right
  // -- for first/last process use MPI_PROC_NULL

  /*
   * Stage 1: get data from the left.
   * Exercise: who are you communicating with?
   */
  sendto=procno+1;
  recvfrom=procno-1;
  if (procno==0) recvfrom=MPI_PROC_NULL;
  if (procno==nprocs-1) sendto=MPI_PROC_NULL;
  /*  
   *
   * Start isend and store the request
   */
  requests.push(
		comm_world.isend(indata[0],sendto));
/**** your code here ****/
  requests.push(
		comm_world.irecv(leftdata,recvfrom));
/**** your code here ****/


  /*
   * Stage 2: data from the right
   * Exercise: who are you communicating with?
   */
  sendto=procno-1;
  recvfrom=procno+1;
  if (procno==0) sendto=MPI_PROC_NULL;
  if (procno==nprocs-1) recvfrom=MPI_PROC_NULL;
  /* 
   * Start isend and store the request
   */
/**** your code here ****/
  requests.push(
		comm_world.isend(indata[N-1],sendto));
/**** your code here ****/
  requests.push(
		comm_world.irecv(rightdata,recvfrom));
/**** your code here ****/

  /*
   * Now make sure all Isend/Irecv operations are completed
   */
/**** your code here ****/
  requests.waitall();
  
  /*
   * Do the averaging operation
   * Note that leftdata==rightdata==0 if not explicitly received.
   */
  for (int i=0; i<N; i++)
    if (i==0)
      outdata[i] = leftdata + indata[i] + indata[i+1];
    else if (i==N-1)
      outdata[i] = indata[i-1] + indata[i] + rightdata;
    else
      outdata[i] = indata[i-1] + indata[i] + indata[i+1];
  
  /*
   * Check correctness of the result:
   * value should be 2 at the end points, 3 everywhere else
   */
  vector<double> answer(N);
  for (int i=0; i<N; i++) {
    if ( (procno==0 && i==0) || (procno==nprocs-1 && i==N-1) ) {
      answer[i] = 2.;
    } else {
      answer[i] = 3.;
    }
  }
  double error_test = array_error(answer,outdata);
  print_final_result(error_test>1.e-5,comm_world);

  return 0;
}
