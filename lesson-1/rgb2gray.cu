/*
 * Convert RGB image to grayscale image
 * I = .299f * R + .587f * G + .114f * B
 *
 * compile:
 * nvcc -o rgb2gray rgb2gray.cu -I ../inc `pkg-config --cflags opencv` `pkg-config --libs opencv`
 */
#include <string>
#include <cstdio>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <opencv2/opencv.hpp>

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;

struct GpuTimer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() {
    cudaEventRecord(start, 0);
  }

  void Stop() {
    cudaEventRecord(stop, 0);
  }

  float Elapsed() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

double get_cpu_time(){
  return (double)clock() / CLOCKS_PER_SEC;
}

size_t numRows();
size_t numCols();

void preProcess(uchar4 **h_rgbaImage, unsigned char **h_greyImage, uchar4 **d_rgbaImage, unsigned char **d_greyImage, const std::string& filename);
void postProcess(const std::string& output_file);

__global__
void rgb2grayGPU_kernel(const uchar4* const rgbaImage, unsigned char* greyImage, size_t numRows, size_t numCols);

void rgb2grayCPU(const uchar4* const rgbaImage, unsigned char* const greyImage, size_t numRows, size_t numCols);
void rgb2grayGPU(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage, unsigned char* const d_greyImage, size_t numRows, size_t numCols);


/*
 * Main routine
 */
int main(int argc, char ** argv) {
  uchar4        *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_greyImage, *d_greyImage;

  std::string input_file;
  std::string output_file;
  std::string processor;
  if (argc == 4) {
    processor = std::string(argv[1]);
    input_file  = std::string(argv[2]);
    output_file = std::string(argv[3]);
  }
  else {
    std::cerr << "Usage: ./rgb2gray -cup/gpu input_file output_file" << std::endl;
    exit(1);
  }

  if (processor == "-gpu") {

    preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

    GpuTimer timer;
    timer.Start();
    rgb2grayGPU(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());
    timer.Stop();
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    printf("\n");
    int err = printf("GPU time: %f msecs.\n", timer.Elapsed());

    if (err < 0) {
      //Couldn't print! Probably the student closed stdout - bad news
      std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
      exit(1);
    }

    //check results and output the grey image
    postProcess(output_file);
  } else if (processor == "-cpu") {
    cv::Mat image;
    image = cv::imread(input_file.c_str(), CV_LOAD_IMAGE_COLOR);
    if (image.empty()) {
      std::cerr << "Couldn't open file: " << input_file << std::endl;
      exit(1);
    }

    cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);
    imageGrey.create(image.rows, image.cols, CV_8UC1);

    double cpu0 = get_cpu_time();
    rgb2grayCPU((uchar4 *)imageRGBA.ptr<unsigned char>(0), imageGrey.ptr<unsigned char>(0), image.rows, image.cols);
    double t = get_cpu_time() - cpu0;
    cv::imwrite(output_file.c_str(), imageGrey);
    printf("CPU time: %f msecs.\n", t * 1000);
  }

  return 0;
}

/*
 * Functions
 */

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

void preProcess(uchar4 **inputImage, unsigned char **greyImage, uchar4 **d_rgbaImage, unsigned char **d_greyImage, const std::string& filename) {
  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));

  cv::Mat image;
  image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

  //allocate memory for the output
  imageGrey.create(image.rows, image.cols, CV_8UC1);

  //This shouldn't ever happen given the way the images are created
  //at least based upon my limited understanding of OpenCV, but better to check
  if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
  *greyImage  = imageGrey.ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
  //allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
  //make sure no memory is left laying around
  checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char)));

  //copy input array to GPU
  checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

  //save as global variables
  d_rgbaImage__ = *d_rgbaImage;
  d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file) {
  const int numPixels = numRows() * numCols();

  //copy the result back to CPU
  checkCudaErrors(cudaMemcpy(imageGrey.ptr<unsigned char>(0), d_greyImage__, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

  //output the image
  cv::imwrite(output_file.c_str(), imageGrey);

  //cleanup
  cudaFree(d_rgbaImage__);
  cudaFree(d_greyImage__);
}


void rgb2grayCPU(const uchar4* const rgbaImage, unsigned char* const greyImage, size_t numRows, size_t numCols) {
  for (size_t r = 0; r < numRows; r++) {
    for (size_t c = 0; c < numCols; c++) {
      uchar4 rgba = rgbaImage[r * numCols + c];
      float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
      greyImage[r * numCols + c] = channelSum;
    }
  }
}

__global__
void rgb2grayGPU_kernel(const uchar4* const rgbaImage, unsigned char* greyImage, size_t numRows, size_t numCols) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = y * numRows + x;

  uchar4 rgba = rgbaImage[idx];
  float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
  greyImage[idx] = channelSum;
}

void rgb2grayGPU(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage, unsigned char* const d_greyImage, size_t numRows, size_t numCols) {
  const dim3 blockSize(32, 32, 1);
  const dim3 gridSize((numRows + 31) / 32, (numCols + 31) / 32, 1);
  rgb2grayGPU_kernel<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
}