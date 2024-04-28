#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <arm_neon.h> //neon 文件

// 定义矩阵的最大规模
const int MAX_SIZE = 4000;

// 生成随机矩阵
void generateMatrix(float** A, int N) {
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++) {
        A[i][i] = 1.0;
        for (int j = 0; j < i; j++) {
            A[i][j] = 0.0;
        }
        for (int j = i + 1; j < N; j++) {
            A[i][j] = rand()%1000;
        }
    }
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] += A[k][j];
                A[i][j]=(int)A[i][j]%1000;
            }
        }
    }
}
// 高斯消去法使用NEON Intrinsics
void gaussEliminationNEON(float** A, int N) {
     for (int k = 0; k < N; k++) {
        float32x4_t vt = vdupq_n_f32(A[k][k]);
        int j;
        for (j=k+1; j + 4 <= N; j += 4) {
            float32x4_t va = vld1q_f32(&A[k][j]);
            va = vdivq_f32(va, vt);
            vst1q_f32(&A[k][j], va);
        }
        for (; j < N; j++) {
            A[k][j] /= A[k][k];
        }
        A[k][k] = 1.0f;

        for (int i = k + 1; i < N; i++) {
            float32x4_t vaik = vdupq_n_f32(A[i][k]);
            for (j = k + 1; j + 4 <= N; j += 4) {
                float32x4_t vakj = vld1q_f32(&A[k][j]);
                float32x4_t vaij = vld1q_f32(&A[i][j]);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(&A[i][j], vaij);
            }
            for (; j < N;j++) {
                A[i][j] -= A[k][j] * A[i][k];
            }
            A[i][k] = 0.0f;
        }
    }

}

int main() {

    std::vector<int> sizes(7);
    sizes.push_back(100);
    sizes.push_back(250);
    sizes.push_back(500);
    sizes.push_back(750);
    sizes.push_back(1000);
    sizes.push_back(2000);
    sizes.push_back(3000);
    sizes.push_back(4000);


    for (int kk = 0; kk < sizes.size(); ++kk) {
        int N = sizes[kk];
        float** A = new float* [N];
        for (int i=0;i<N;i++){
            A[i]=new float[N];
        }
        // 生成测试矩阵
        generateMatrix(A, N);

        struct timeval start_time, end_time;
        gettimeofday(&start_time, NULL);

        // 进行高斯消元
        gaussEliminationNEON(A, N);

        gettimeofday(&end_time, NULL);
        long long int diff_usec = (end_time.tv_sec - start_time.tv_sec) * 1000000LL + end_time.tv_usec - start_time.tv_usec;

        double elapsed = static_cast<double>(diff_usec) / 1e6; // 转换为秒

        std::cout << "Matrix size: " << N << ", Time taken: " << elapsed << " seconds.\n";

        for(int i=0;i<N;i++){
            delete[]A[i];
        }
        delete[] A;
    }

    return 0;
}
