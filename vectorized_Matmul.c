#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <smmintrin.h>

#define MSIZE 4 //Size of square matrix
#define VROW 4  //rows of vector
#define VCOLUMN 4 //columns of vector

#define XMM_ALIGNMENT_BYTES 16
static float mat_a[MSIZE][MSIZE] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float mat_b[VROW][VCOLUMN] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float mat_c[VROW][VCOLUMN] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float mat_ref[VROW][VCOLUMN] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));

static void init_matrices()
{
        int i, j;

        for (i = 0; i < MSIZE; i++) {
                for (j = 0; j < MSIZE; j++) {
                        mat_a[i][j] = (((i * (j)) / MSIZE) & 0x0F) * 0x1P-4F;
                }
        }

        for (i = 0; i < VROW; i++) {
                for (j = 0; j < VCOLUMN; j++) {
                        mat_b[i][j] = (((i * (j))*2 / MSIZE) & 0x0F) * 0x1P-4F;
                }
        }

        memset(mat_c, 0, sizeof(mat_c));
        memset(mat_ref, 0, sizeof(mat_ref));
}


static void matrixMul(){

  for(int i = 0; i < MSIZE;i++){
    for(int j=0;j < VCOLUMN;j++ ){
      for(int k =0 ;  k < MSIZE ;k++ ){
        mat_ref[i][j] += mat_a[i][k]*mat_b[j][k];

      }
    }
  }
}

static void unroll_matrixMul(){

  for(int i = 0; i < MSIZE;i++){
    for(int j=0;j < VCOLUMN;j++ ){
      for(int k =0 ;  k < MSIZE ;k+=4 ){
        mat_ref[i][j] += mat_a[i][k]*mat_b[j][k]      +
                         mat_a[i][k+1]*mat_b[j][k+1]  +
                         mat_a[i][k+2]*mat_b[j][k+2]  +
                         mat_a[i][k+3]*mat_b[j][k+3];

      }
    }
  }
}

static void matrixMul_SSE(){
  __m128 row1;
  __m128 column1;
  volatile __m128 zero = _mm_setzero_ps();
  __m128 acc1;

  for(int i = 0; i < MSIZE; i++) // Row
  {
      for(int j = 0; j < VCOLUMN; j++) // Column
      {
      acc1 = _mm_setzero_ps();
      for (int k=0; k < VROW; k+=4) {
        row1 = _mm_load_ps(&mat_a[i][k]);
        column1 =_mm_load_ps(&mat_b[j][k]);
        acc1 = _mm_add_ps(_mm_mul_ps(row1, column1), acc1);
      }
      acc1 = _mm_hadd_ps(acc1, zero);
      acc1 = _mm_hadd_ps(acc1, zero);
      mat_c[i][j] = _mm_cvtss_f32(acc1);
      }

  }

}

static void verify_result_con_vs_sse()
{
        float error_sum;
        int i;

        error_sum = 0;

        for (i = 0; i < MSIZE; i++) {
          for(int j=0;j<VCOLUMN;j++){
                printf("%f ",mat_ref[i][j] );
              }
            printf("\n");
        }

printf("==================================================\n" );
        for (i = 0; i < MSIZE; i++) {
          for(int j=0;j<VCOLUMN;j++){
              printf("%f ",mat_c[i][j] );
              }
                      printf("\n");
        }
}


int main(int argc, char const *argv[]) {

  init_matrices();
  matrixMul();
  matrixMul_SSE();
  verify_result_con_vs_sse();

  return 0;
}
