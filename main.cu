#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_ITER 1000
#define BLOCK_SIZE 32
#define WIDTH 1920
#define HEIGHT 1920
#define CHECK_CUDA(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

__device__ int mandelbrot(double real, double imaginary) {
    double zr = 0.;
    double zi = 0.;

    int iter = 0;
    for (iter = 0; iter < MAX_ITER; iter++) {
        double temp = zr*zr - zi*zi + real;
        zi = 2.0 * zr * zi + imaginary;
        zr = temp;

        if (zr*zr + zi*zi > 4.) break;
    }

    return iter;
}

__global__ void render_image(unsigned char* image, float xmax, float xmin, float ymax, float ymin) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;


    double real =      xmin + (x / (double)WIDTH) * (xmax - xmin);
    double imaginary = ymin + (y / (double)HEIGHT) * (ymax - ymin);

    int iterations = mandelbrot(real, imaginary);

    image[y * WIDTH + x] = (unsigned char)(255 * iterations / MAX_ITER);
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

    render_image<<<grid, block>>>(d_image, xmax, xmin, ymax, ymin);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaMemcpy(h_image, d_image, sizeof(unsigned char) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);

    save_pgm("test_2.pgm", h_image);

    printf("\n\nDONE\n\n");
    return 0;
}
