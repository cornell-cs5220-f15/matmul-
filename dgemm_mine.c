const char* dgemm_desc = "My super awesome dgemm.";

#ifndef block_size
#define block_size ((int) 64)
#endif

#ifndef rect_length
#define rect_length ((int) 256)
#endif

// Blocked type
void square_dgemm(const int M, const double* restrict A, const double* restrict B, double* restrict C)
{
  int I,J,K,i,j,k,i_end,j_end,k_end;
  int Aind,Bind,Cind,Aindf,Bindf,Cindf;
  double BRHS;
  
  const int n_rect = M/rect_length + (M%rect_length? 1 : 0);
  const int n_blocks = M / block_size + (M%block_size? 1 : 0);
  
  //Block Loop
  for(J=0; J<n_blocks; ++J) {
    for(K=0; K<n_blocks; ++K) {
      for(I=0; I<n_rect; ++I){
        j_end = ((J+1)*block_size < M? block_size : (M-(J*block_size)));
        k_end = ((K+1)*block_size < M? block_size : (M-(K*block_size)));
        i_end = ((I+1)*rect_length < M? rect_length : (M-(I*rect_length)));
        Aind = (I*rect_length)+(K*block_size)*M;
        Bind = (K*block_size)+(J*block_size)*M;
        Cind = (I*rect_length)+(J*block_size)*M;
        //Inner block loop
        for(j=0;j< j_end; ++j) {
          Bindf = Bind + j*M;
          Cindf = Cind+j*M;
          for(k=0; k<k_end; ++k) {
            Aindf = Aind+k*M;
            BRHS = B[Bindf+k];
            for(i=0; i<i_end; ++i) {
              C[Cindf+i] += A[Aindf+i]*BRHS;
              }
            }
         }
      }
    }
  }
}
