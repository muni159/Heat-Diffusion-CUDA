# HeatDiffusion-CUDA.MPI
Simulated 1D heat diffusion with MPI and Simulated 2D &amp; 3D heat diffusion with CUDA.
To excecute the codes, make sure you have correctly installed openmpi and Cuda.

Version of cmake must be above 2.8 (Recommended 3.9.1).


# How to run
For 1D codes, after doing cmake, execute:

$mpirun -np [core number]  heat1D  [left fixed temperture]  [right fixed temperture]  [grids]  [time steps] 



For 2D3D codes, after doing cmake, execute:

$./heat2D3D  [configFile.txt]
