#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <smmintrin.h>

#define MSIZE 4 //Size of square matrix
#define VROW 4  //rows of vector

#define XMM_ALIGNMENT_BYTES 16
static float mat_a[MSIZE][MSIZE] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float vec_b[VROW] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float vec_c[VROW] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));
static float vec_ref[VROW] __attribute__((aligned (XMM_ALIGNMENT_BYTES)));

static void init()
{
        int i, j;

        for (i = 0; i < MSIZE; i++) {
                for (j = 0; j < MSIZE; j++) {
                        mat_a[i][j] = (((i * (j)) / MSIZE) & 0x0F) * 0x1P-4F;
                }
        }

        for (i = 0; i < VROW; i++) {
                        vec_b[i] = (((i * (j))*2 / MSIZE) & 0x0F) * 0x1P-4F;
        }

        memset(vec_c, 0, sizeof(vec_c));
        memset(vec_ref, 0, sizeof(vec_ref));
}


static void matrixMul(){

  for(int i = 0; i < MSIZE;i++){
    for(int j=0;j < VROW;j++ ){
        vec_ref[i] += mat_a[i][j]*vec_b[j];
    }
  }
}

static void unroll_matrixMul(){

  for(int i = 0; i < MSIZE;i++){
    for(int j=0;j < VROW;j+=4 ){
        vec_ref[i] += mat_a[i][j]*vec_b[j]      +
                         mat_a[i][j+1]*vec_b[j+1]  +
                         mat_a[i][j+2]*vec_b[j+2]  +
                         mat_a[i][j+3]*vec_b[j+3];
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
      printf("adf\n");
      acc1 = _mm_setzero_ps();
      for(int j = 0; j < VROW; j+=4) // Column
      {
        row1 = _mm_load_ps(&mat_a[i][j]);
        column1 =_mm_load_ps(&vec_b[j]);
        acc1 = _mm_add_ps(_mm_mul_ps(row1, column1), acc1);
      }
      acc1 = _mm_hadd_ps(acc1, zero);
      acc1 = _mm_hadd_ps(acc1, zero);
      vec_c[i] = _mm_cvtss_f32(acc1);

  }

}

static void verify_result_con_vs_sse()
{
        float error_sum;
        int i;

        error_sum = 0;

        for (i = 0; i < MSIZE; i++) {
                printf("%f \n",vec_ref[i] );
        }

printf("==================================================\n" );
        for (i = 0; i < MSIZE; i++) {
              printf("%f\n",vec_c[i] );
        }
}


int main(int argc, char const *argv[]) {

  init();

  matrixMul();

  matrixMul_SSE();
  verify_result_con_vs_sse();

  return 0;
}
