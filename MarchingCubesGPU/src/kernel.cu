
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
__global__ void buildBasePyramid(const int *triTableSizes, const float *grid, int gridSize, int *oBasePyramid) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int ix = index / (gridSize * gridSize);
	int iy = (index / gridSize) % gridSize;
	int iz = index % gridSize;
	int indexBase = 0; // TODOOOOOOOOO
	int indexVoxel = 0;
	int baseSize = gridSize - 1;
	if (ix < baseSize && iy < baseSize && iz < baseSize) {
		indexVoxel |= (grid[index] > 0) * 1;
		indexVoxel |= (grid[index + gridSize * gridSize] > 0) * 2;
		indexVoxel |= (grid[index + gridSize * gridSize + gridSize] > 0) * 4;
		indexVoxel |= (grid[index + gridSize] > 0) * 8;
		indexVoxel |= (grid[index + gridSize + 1] > 0) * 16;
		indexVoxel |= (grid[index + gridSize * gridSize + gridSize + 1] > 0) * 32;
		indexVoxel |= (grid[index + gridSize * gridSize + 1] > 0) * 64;
		indexVoxel |= (grid[index + 1] > 0) * 128;
		oBasePyramid[ix * baseSize * baseSize + iy * baseSize + iz] = triTableSizes[indexVoxel];
	}
}


__global__ void buildPyramidStep(int *oBasePyramid, int previousGridOffset, int currentGridOffset, int currentGridSize) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int ix = index / (currentGridSize * currentGridSize);
	int iy = (index / currentGridSize) % currentGridSize;
	int iz = index % currentGridSize;
	if (ix < currentGridSize && iy < currentGridSize && iz < currentGridSize) {
		int previousGridSize = currentGridSize * 2;
		//int previousIndex = previousGridOffset + index;
		int previousIndex = previousGridOffset + 2 * (ix * previousGridSize * previousGridSize + iy * previousGridSize + iz);
		index += currentGridOffset;
		oBasePyramid[index] = oBasePyramid[previousIndex];

		oBasePyramid[index] += oBasePyramid[previousIndex + 1];
		oBasePyramid[index] += oBasePyramid[previousIndex + previousGridSize];
		oBasePyramid[index] += oBasePyramid[previousIndex + previousGridSize * previousGridSize];

		oBasePyramid[index] += oBasePyramid[previousIndex + previousGridSize * previousGridSize + 1];
		oBasePyramid[index] += oBasePyramid[previousIndex + previousGridSize * previousGridSize + previousGridSize];
		oBasePyramid[index] += oBasePyramid[previousIndex + 1 + previousGridSize];

		oBasePyramid[index] += oBasePyramid[previousIndex + previousGridSize * previousGridSize + previousGridSize + 1];
	}

}

#define CUDACHECKMSG(message, command) \
	if (cudaSuccess != (cudaStatus =(command))) { std::cout << (message) << std::endl; goto EndAux1;}

#define CUDACHECK(command)\
	CUDACHECKMSG("Cuda check failed", (command))


void writeMiddle(const char *file, int *tab, int offset, int size) {
	writePGMToFileInt(file, size, size, &tab[offset + size * size * (size / 2)]);
}



cudaError_t computeTrianglesAux1(std::vector<float> &triangles) {

	// declarations
	std::cout << "yoo" << std::endl;
	cudaError_t cudaStatus;
	float *functionValuesGPU = 0;
	int *triTableSizesGPU = 0;
	int *histoPyramidBaseGPU = 0;
	int blockSize = 512;
	int nbBlocks = FCT_TOTAL_GRID_SIZE / blockSize;
	std::cout << "block size : " << blockSize << std::endl;
	std::cout << "nb blocks : " << nbBlocks << std::endl;
	dim3 cudaBlockSize(blockSize);
	dim3 cudaGridSize(nbBlocks);

	// allocations
	CUDACHECKMSG("functionValuesGPU allocation failed", cudaMalloc((void**)&functionValuesGPU, FCT_TOTAL_GRID_SIZE * sizeof(float)));
	CUDACHECKMSG("triTableSizesGPU allocation failed", cudaMalloc((void**)&triTableSizesGPU, 256 * sizeof(int)));
	CUDACHECKMSG("histoPyramidBaseGPU allocation failed", cudaMalloc((void**)&histoPyramidBaseGPU, 2 * PYRAMID_TOTAL_GRID_SIZE * sizeof(int)));

	// fill values grid
	computeFonction <<<cudaGridSize, cudaBlockSize >>>(functionValuesGPU, FCT_GRID_SIZE);
	CUDACHECKMSG("computeFonction failed", cudaGetLastError())

	// fill tritablesizes, could be parallel with previous things
	CUDACHECKMSG("cannot fill tritablesizes", cudaMemcpy(triTableSizesGPU, triTableSizes, 256 * sizeof(int), cudaMemcpyHostToDevice));

	// compute base pyramid
	buildBasePyramid << <cudaGridSize, cudaBlockSize >> >(triTableSizesGPU, functionValuesGPU, FCT_GRID_SIZE, histoPyramidBaseGPU);
	CUDACHECKMSG("buildBasePyramid failed", cudaGetLastError())


	buildPyramidStep << <cudaGridSize, cudaBlockSize >> >(histoPyramidBaseGPU, 0, PYRAMID_TOTAL_GRID_SIZE, PYRAMID_GRID_SIZE / 2);
	CUDACHECKMSG("buildBasePyramid failed", cudaGetLastError())
	

	/*this is only for debug*/
	{
		float * functionValuesCPU = new float[FCT_TOTAL_GRID_SIZE];
		int * histoPyramidBaseCPU = new int[PYRAMID_TOTAL_GRID_SIZE * 2];
		if (!functionValuesCPU) {
			std::cout << "Cannot allocate functionValuesCPU" << std::endl;
		}
		if (!histoPyramidBaseCPU) {
			std::cout << "Cannot allocate histoPyramidBaseGPU" << std::endl;
		}
		CUDACHECKMSG("debug 1  failed", cudaMemcpy(functionValuesCPU, functionValuesGPU, FCT_TOTAL_GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
		writePGMToFileFloat("c:\\temp\\valuesGridMiddle.pgm", FCT_GRID_SIZE, FCT_GRID_SIZE, &functionValuesCPU[(FCT_GRID_SIZE * FCT_GRID_SIZE) * (FCT_GRID_SIZE / 2)]);
		delete[] functionValuesCPU; 
		CUDACHECKMSG("debug 2  failed", cudaMemcpy(histoPyramidBaseCPU, histoPyramidBaseGPU, 2 * PYRAMID_TOTAL_GRID_SIZE * sizeof(int), cudaMemcpyDeviceToHost));
		writeMiddle("c:\\temp\\histoMiddle.pgm", histoPyramidBaseCPU, 0, PYRAMID_GRID_SIZE);
		writeMiddle("c:\\temp\\histoMiddle1.pgm", histoPyramidBaseCPU, PYRAMID_TOTAL_GRID_SIZE, PYRAMID_GRID_SIZE / 2);
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
