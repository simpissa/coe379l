#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Stampede3 SKX nodes
#
#   *** MPI Job in SKX Queue ***
#
# Last revised: 23 April 2024
#
# Notes:
#
#   -- Launch this script by executing
#      "sbatch skx.mpi.slurm" on Stampede3 login node.
#
#   -- Use ibrun to launch MPI codes on TACC systems.
#      Do not use mpirun or mpiexec.
#
#   -- Max recommended MPI ranks per SKX node: 48
#      (start small, increase gradually).
#
#   -- If you're running out of memory, try running
#      fewer tasks per node to give each task more memory.
#
#----------------------------------------------------

#SBATCH -J transpose           # Job name
#SBATCH -o transpose.o%j       # Name of stdout output file
#SBATCH -e transpose.e%j       # Name of stderr error file
#SBATCH -p skx             # Queue (partition) name
#SBATCH -N 4               # Total # of nodes
#SBATCH -n 16              # Total # of mpi tasks
#SBATCH -t 00:01:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=mzhao12@tacc.utexas.edu
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A transpose       # Allocation name (req'd if you have more than 1)

# Other commands must follow all #SBATCH directives...:

module list
pwd
date

# Launch MPI code...
for x in 1 4 16 ; do ibrun -n $x ./transpose ; done

# ibrun ./rightsend         # Use ibrun instead of mpirun or mpiexec
# for x in 5 10 20 40 ; do ibrun -n $x ./rightsend ; done
                                                        
