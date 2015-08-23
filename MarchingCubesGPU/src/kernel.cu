
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
#include <vector>
#include <iostream>
#include "../../Utils/src/outputs.hpp"
#include "mcgpuconstants.hpp"

#include <stdio.h>
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

/**
*	Fill grid with the isosurface function
*   grid is a grid[gridSize][gridSize][gridSize]
*/
__global__ void computeFonction(float *grid, int gridSize)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < gridSize * gridSize * gridSize) {
		int ix = index / (gridSize * gridSize);
		int iy = (index / gridSize) % gridSize;
		int iz = index % gridSize;
		float x = (2 * ix / float(gridSize)) - 1.0;
		float y = (2 * iy / float(gridSize)) - 1.0;
		float z = (2 * iz / float(gridSize)) - 1.0;
		grid[index] = x * x + y * y + z * z - 0.5;
	}
}

/**
*	look at the 8 corners of the voxel and deduce how many triangles are needed
*/
__global__ void buildBasePyramid(const int *triTableSizes, const float *grid, int gridSize, char *oBasePyramid) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int ix = index / (gridSize * gridSize);
	int iy = (index / gridSize) % gridSize;
	int iz = index % gridSize;
	int indexVoxel = 0;
	if (ix < gridSize - 1 && iy < gridSize - 1 && iz < gridSize - 1) {
		indexVoxel |= (grid[index] > 0) * 1;
		indexVoxel |= (grid[index + gridSize * gridSize] > 0) * 2;
		indexVoxel |= (grid[index + gridSize * gridSize + gridSize] > 0) * 4;
		indexVoxel |= (grid[index + gridSize] > 0) * 8;
		indexVoxel |= (grid[index + gridSize + 1] > 0) * 16;
		indexVoxel |= (grid[index + gridSize * gridSize + gridSize + 1] > 0) * 32;
		indexVoxel |= (grid[index + gridSize * gridSize + 1] > 0) * 64;
		indexVoxel |= (grid[index + 1] > 0) * 128;
		oBasePyramid[index] = triTableSizes[indexVoxel];
	}
}

#define CUDACHECKMSG(message, command) \
	if (cudaSuccess != (cudaStatus =(command))) { std::cout << (message) << std::endl; goto EndAux1;}

#define CUDACHECK(command)\
	CUDACHECKMSG("Cuda check failed", (command))





cudaError_t computeTrianglesAux1(std::vector<float> &triangles) {

	// declarations
	std::cout << "yoo" << std::endl;
	cudaError_t cudaStatus;
	float *functionValuesGPU = 0;
	int *triTableSizesGPU = 0;
	char *histoPyramidBaseGPU = 0;
	//int *histoPyramidNoBaseGPU = 0;
	int blockSize = 512;
	int nbBlocks = FCT_TOTAL_GRID_SIZE / blockSize;
	std::cout << "block size : " << blockSize << std::endl;
	std::cout << "nb blocks : " << nbBlocks << std::endl;
	dim3 cudaBlockSize(blockSize);
	dim3 cudaGridSize(nbBlocks);

	// allocations
	CUDACHECKMSG("functionValuesGPU allocation failed", cudaMalloc((void**)&functionValuesGPU, FCT_TOTAL_GRID_SIZE * sizeof(float)));
	CUDACHECKMSG("triTableSizesGPU allocation failed", cudaMalloc((void**)&triTableSizesGPU, 256 * sizeof(int)));
	CUDACHECKMSG("histoPyramidBaseGPU allocation failed", cudaMalloc((void**)&histoPyramidBaseGPU, FCT_TOTAL_GRID_SIZE * sizeof(char)));
	//CUDACHECKMSG("histoPyramidNoBaseGPU allocation failed", cudaMalloc((void**)&histoPyramidBaseGPU, FCT_TOTAL_GRID_SIZE * sizeof(int)));

	// fill values grid
	computeFonction <<<cudaGridSize, cudaBlockSize >>>(functionValuesGPU, FCT_GRID_SIZE);
	CUDACHECKMSG("computeFonction failed", cudaGetLastError())

	// fill tritablesizes, could be parallel with previous things
	CUDACHECKMSG("cannot fill tritablesizes", cudaMemcpy(triTableSizesGPU, triTableSizes, 256 * sizeof(int), cudaMemcpyHostToDevice));

	// compute base pyramid
	buildBasePyramid << <cudaGridSize, cudaBlockSize >> >(triTableSizesGPU, functionValuesGPU, FCT_GRID_SIZE, histoPyramidBaseGPU);
	CUDACHECKMSG("buildBasePyramid failed", cudaGetLastError())

	/*this is only for debug*/
	{
		float * functionValuesCPU = new float[FCT_TOTAL_GRID_SIZE];
		char * histoPyramidBaseCPU = new char[FCT_TOTAL_GRID_SIZE];
		if (!functionValuesCPU) {
			std::cout << "Cannot allocate functionValuesCPU" << std::endl;
		}
		if (!histoPyramidBaseCPU) {
			std::cout << "Cannot allocate histoPyramidBaseGPU" << std::endl;
		}
		CUDACHECKMSG("debug failed", cudaMemcpy(functionValuesCPU, functionValuesGPU, FCT_TOTAL_GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
		writePGMToFileFloat("c:\\temp\\valuesGridMiddle.pgm", FCT_GRID_SIZE, FCT_GRID_SIZE, &functionValuesCPU[FCT_TOTAL_GRID_SIZE / 2]);
		delete[] functionValuesCPU;
		CUDACHECKMSG("debug failed", cudaMemcpy(histoPyramidBaseCPU, histoPyramidBaseGPU, FCT_TOTAL_GRID_SIZE * sizeof(char), cudaMemcpyDeviceToHost));
		writePGMToFileChar("c:\\temp\\histoMiddle.pgm", FCT_GRID_SIZE, FCT_GRID_SIZE, &histoPyramidBaseCPU[FCT_TOTAL_GRID_SIZE / 2]);
		delete[] histoPyramidBaseCPU;
	}



	// the end
EndAux1:
	if (cudaStatus != cudaSuccess) {
		std::cout << "Error!" << std::endl;
	}
	cudaFree(functionValuesGPU);
	cudaFree(triTableSizesGPU);
	cudaFree(histoPyramidBaseGPU);

	return cudaStatus;
}
cudaError_t computeTrianglesAux2(std::vector<float> &triangles) {
	triangles.push_back(1.0);
	triangles.push_back(2.0);
	triangles.push_back(0.5);
	triangles.push_back(1.0);
	triangles.push_back(0.7);
	triangles.push_back(0.0);
	triangles.push_back(0.0);
	triangles.push_back(0.3);
	triangles.push_back(1.0);
	return cudaSuccess;
}

void computeTriangles(std::vector<float> &triangles)
{
	std::cout << "computeTriangles begin" << std::endl;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	cudaStatus = computeTrianglesAux1(triangles);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "computeTrianglesAux2 failed!");
		return;
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return;
	}

	std::cout << "computeTriangles end ok" << std::endl;
}
