# Box Filter NPP CUDA Sample

Hi my name is Shashank. This project demonstrates how to use the NPP (NVIDIA Performance Primitives) library's `FilterBox` function to perform a box filter on images. The box filter is a simple averaging filter that is commonly used in image processing to blur images.

## Minimum Specifications

- **CUDA Compute Capability**: SM 2.0 or higher
- **NVIDIA GPU**: Compatible with CUDA compute capability 2.0 or higher

## Key Concepts

- **Performance Strategies**: Leveraging NPP for optimized performance in image processing tasks.
- **Image Processing**: Applying a box filter to an image to achieve blurring effects.
- **NPP Library**: Using NVIDIA's performance primitives for efficient image processing.

## Prerequisites

- CUDA Toolkit installed on your system.
- NPP library included in your CUDA installation.
- A compatible NVIDIA GPU with CUDA support.

## Building the Project

1. **Clone the Repository**

   If you haven’t already, clone the repository containing the source code:

   ```bash
   git clone <repository-url>
   cd <repository-directory>


2. **Navigate to the Project Directory**
Ensure you are in the directory containing the Makefile and src directory.
cd <project-directory>

3. **This will compile the source code and generate the executable.**
    make

4. **Running the Project**
    make run
5. **Cleaning Up**
    make clean
