#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define BLOCK_SIZE 32
#define WIDTH 7680
#define HEIGHT 7680

#define ZMAX 4.
#define MAX_ITER 100
#define CREAL -0.5251993
#define CIMAGINARY -0.5251993

#define CHECK_CUDA(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

double getTimeMicroseconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

__device__ int julia(double real, double imaginary, int n, double cr, double ci, double max) {
    int iter = 0;
    for (iter = 0; iter < n; iter++) {
        double temp = real*real - imaginary*imaginary + cr;
        imaginary = 2.0 * real * imaginary + ci;
        real = temp;

        if (real*real + imaginary*imaginary > max) break;
    }

    return iter;
}

__global__ void render_image(unsigned char* image, float xmax, float xmin, float ymax, float ymin) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;


    double real =      xmin + (x / (double)WIDTH) * (xmax - xmin);
    // here this strange formula is needed because the top left corder of the image
    // is considered the (0, 0) origin
    double imaginary = ymin + ((HEIGHT - 1 - y) / (double)HEIGHT) * (ymax - ymin);

    int iterations = julia(real, imaginary, MAX_ITER, CREAL, CIMAGINARY, ZMAX);

    double intensity = atan(0.1 * iterations);
    image[y * WIDTH + x] = (int)(intensity * 255);
}

void save_pgm(const char *filename, unsigned char *data) {
	FILE *fp = fopen(filename, "wb");
	fprintf(fp, "P5\n%d %d\n255\n", (int)WIDTH, (int)HEIGHT);
	fwrite(data, 1, WIDTH * HEIGHT, fp);
	fclose(fp);
}


int main() {
    unsigned char* d_image, *h_image;
    h_image = (unsigned char*)malloc(WIDTH * HEIGHT);
    cudaMalloc(&d_image, WIDTH * HEIGHT);

    float xmax = 2.;
    float xmin = -2.;

    float ymax = 2.;
    float ymin = -2.;

    dim3 grid((WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    double start = getTimeMicroseconds();
    render_image<<<grid, block>>>(d_image, xmax, xmin, ymax, ymin);
    CHECK_CUDA(cudaDeviceSynchronize());
    double end = getTimeMicroseconds();

    cudaMemcpy(h_image, d_image, sizeof(unsigned char) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    save_pgm("output.pgm", h_image);

    printf("\n\nDONE in %lf microseconds \n\n", end - start);
    return 0;
}
