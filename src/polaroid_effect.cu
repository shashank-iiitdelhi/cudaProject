#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sys/stat.h>

// CUDA kernel for tinting, temperature, and saturation adjustments
__global__ void polaroidKernel(unsigned char* img, int width, int height, float3 tint, float temperature, float saturation) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3; // RGB has 3 channels

        // Apply temperature adjustment
        float tempRed = img[idx] + temperature;
        float tempGreen = img[idx + 1];
        float tempBlue = img[idx + 2] - temperature;

        // Apply saturation adjustment
        float gray = 0.2989f * tempRed + 0.5870f * tempGreen + 0.1140f * tempBlue;
        float newRed = gray + (tempRed - gray) * saturation;
        float newGreen = gray + (tempGreen - gray) * saturation;
        float newBlue = gray + (tempBlue - gray) * saturation;

        // Apply tint
        newRed = newRed * tint.x;
        newGreen = newGreen * tint.y;
        newBlue = newBlue * tint.z;

        // Clip values to the valid range [0, 255] or use nearest valid value
        newRed = min(255.0f, max(0.0f, newRed));
        newGreen = min(255.0f, max(0.0f, newGreen));
        newBlue = min(255.0f, max(0.0f, newBlue));

        // Apply vignette effect
        float dx = (2.0f * x / width - 1.0f);
        float dy = (2.0f * y / height - 1.0f);
        float dist = sqrtf(dx * dx + dy * dy);
        float vignette = 1.0f - dist * 0.5f;

        // Ensure colors are not altered excessively in dark areas
        float lum = 0.2989f * img[idx] + 0.5870f * img[idx + 1] + 0.1140f * img[idx + 2];
        float lumFactor = lum / 255.0f;
        newRed = min(255.0f, newRed * (0.5f + 0.5f * lumFactor));
        newGreen = min(255.0f, newGreen * (0.5f + 0.5f * lumFactor));
        newBlue = min(255.0f, newBlue * (0.5f + 0.5f * lumFactor));

        img[idx] = static_cast<unsigned char>(newRed * vignette);
        img[idx + 1] = static_cast<unsigned char>(newGreen * vignette);
        img[idx + 2] = static_cast<unsigned char>(newBlue * vignette);
    }
}


void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void loadPPM(const std::string& filename, unsigned char*& data, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file " + filename);
    }

    std::string header;
    file >> header;
    if (header != "P6") {
        throw std::runtime_error("Invalid PPM file: " + filename);
    }

    file >> width >> height;
    int maxVal;
    file >> maxVal;
    file.ignore(); // Skip the newline character after maxVal

    int dataSize = width * height * 3;
    data = new unsigned char[dataSize];
    file.read(reinterpret_cast<char*>(data), dataSize);
    file.close();
}

void savePPM(const std::string& filename, unsigned char* data, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file " + filename);
    }

    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<char*>(data), width * height * 3);

    file.close();
}

int main(int argc, char* argv[]) {
    printf("%s Starting...\n\n", argv[0]);

    try {
        std::vector<std::string> inputFiles = {"images/img1.ppm", "images/img2.ppm", "images/img3.ppm", "images/img4.ppm", "images/img5.ppm","images/img6.ppm","images/img7.ppm","images/img8.ppm"};

        std::string outputDir = "processed_images";
        // Create the output directory
        #if defined(_WIN32)
            _mkdir(outputDir.c_str());
        #else 
            mkdir(outputDir.c_str(), 0755);
        #endif

        for (const auto& sFilename : inputFiles) {
            unsigned char* h_rgb = nullptr;
            int width, height;
            loadPPM(sFilename, h_rgb, width, height);

            unsigned char* d_rgb;
            checkCudaErrors(cudaMalloc(&d_rgb, width * height * 3 * sizeof(unsigned char)));
            checkCudaErrors(cudaMemcpy(d_rgb, h_rgb, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));

            dim3 blockSize(16, 16);
            dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

            // Applying a Polaroid effect with a slight yellow tint, temperature increase, and saturation adjustment
            polaroidKernel<<<gridSize, blockSize>>>(d_rgb, width, height, make_float3(1.2f, 1.1f, 0.9f), 10.0f, 1.3f);
            cudaDeviceSynchronize();

            checkCudaErrors(cudaMemcpy(h_rgb, d_rgb, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
            cudaFree(d_rgb);

            std::string outputFilename = outputDir + "/" + sFilename.substr(7, sFilename.find_last_of('.') - 7) + "_polaroid.ppm";
            savePPM(outputFilename, h_rgb, width, height);
            std::cout << "Saved image: " << outputFilename << std::endl;

            delete[] h_rgb;
        }

        exit(EXIT_SUCCESS);
    } catch (std::exception& e) {
        std::cerr << "Program error! The following exception occurred: \n" << e.what() << std::endl;
        exit(EXIT_FAILURE);
    } catch (...) {
        std::cerr << "Program error! An unknown type of exception occurred. \n";
        exit(EXIT_FAILURE);
    }

    return 0;
}
