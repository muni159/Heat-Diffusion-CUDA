#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include "mpi.h"

using namespace std;

int main(int argc,char**argv) {
   int rank, size;   
   double send_left, send_right, recv_left, recv_right;
   double T1temp = strtod(argv[1], NULL);
   double T2temp = strtod(argv[2], NULL);
   int grid = atoi (argv[3]);
   int steps = atoi (argv[4]);

   // --------------------------- MPI Start --------------------------------

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   if (grid<size){
      size = grid;
   }
   // --------------------------- initial array ----------------------------
   int length;
   if (rank <= grid%size-1){
      length = (grid/size)+3;
     }
   else{
      length = (grid/size)+2;
     }
   double *meo = new double[length];
   double *tmp = new double[length];
   for (int m=0; m<length; m++){
         meo[m] = 0;
         tmp[m] = 0;
      }
   // --------------------------- begin loop --------------------------------
   if (rank == 0){
      if (size == 1){
         meo[length-1] = T2temp;
      }
      meo[0] = T1temp;
      for (int i=0; i<steps; i++){
         //cout<< "Rank0" << endl;
         send_right = meo[length-2];
      if (size>1){
         MPI_Send(&send_right, 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
         MPI_Recv(&recv_right, 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         meo[length-1] = recv_right;
      }
         for (int j=1; j<length-1; j++){
            tmp[j] = 0.25*meo[j-1]+0.25*meo[j+1]+0.5*meo[j];
         }
         for (int j=1; j<length-1; j++){
            meo[j] = tmp[j];
         }
      }
      // for (int m=1; m<length-1; m++){
      //    cout<< " " << meo[m] << " " << endl;
      // }
      MPI_Send(&meo[1], length-2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
   }
   else if (rank == size-1){
      meo[length-1] = T2temp;
      for (int i=0; i<steps; i++){
         //cout<< "Rank" << rank << endl;
         send_left = meo[1];
         MPI_Send(&send_left, 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
         MPI_Recv(&recv_left, 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         meo[0] = recv_left;
         for (int j=1; j<length-1; j++){
            tmp[j] = 0.25*meo[j-1]+0.25*meo[j+1]+0.5*meo[j];
         }
         for (int j=1; j<length-1; j++){
            meo[j] = tmp[j];
         }
      }
      // for (int m=1; m<length-1; m++){
      //    cout<< " " << meo[m] << " " << endl;
      // }
      MPI_Send(&meo[1], length-2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
   }
   else if(rank<size){
      for (int i=0; i<steps; i++){
         //cout<< "Rank" << rank << endl;
         send_left = meo[1];
         send_right = meo[length-2];
         MPI_Send(&send_left, 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
         MPI_Send(&send_right, 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
         MPI_Recv(&recv_left, 1, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         MPI_Recv(&recv_right, 1, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         meo[0] = recv_left;
         meo[length-1] = recv_right;
         for (int j=1; j<length-1; j++){
            tmp[j] = 0.25*meo[j-1]+0.25*meo[j+1]+0.5*meo[j];
         }
         for (int j=1; j<length-1; j++){
            meo[j] = tmp[j];
         }
      }
      // for (int m=1; m<length-1; m++){
      //    cout<< " " << meo[m] << " " << endl;
      // }
      MPI_Send(&meo[1], length-2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
   }
   

   //------------------------------- result collection -------------------------------
   if (rank == 0) {
      //MPI_Recv(&meo, (grid/size)+2+tag, MPI_DOUBLE, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      double *res = new double[grid];
      int index = 0;
      int tag = 1;
      int tlength = length;
      int point = 0;
      for (int k=0; k<size; k++){
         if (k <= grid%size-1)
            tlength = (grid/size)+3;
         else
            tlength = (grid/size)+2;
         MPI_Recv(&res[point], tlength-2, MPI_DOUBLE, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         point += tlength-2;
      }

      //----------- store result -------------
      for (int m=0; m<grid; m++){
         if (m!=grid-1)
            cout<< res[m] << ", ";
         else
            cout<< res[m];
      }
      cout << "\n";

      std::ofstream myfile;
      myfile.open ("heat1Doutput.csv");
      for (int m=0; m<grid; m++){
         if (m!=grid-1)
            myfile<< res[m] << ", ";
         else
            myfile<< res[m];
      }
      myfile.close();
   }
   MPI_Finalize();

   return 0;
} 