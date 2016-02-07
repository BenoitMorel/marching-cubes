
#include "device_launch_parameters.h"
#include <vector>
#include <iostream>
#include <sstream>



#include "kernel.cuh"
#include "../../Utils/src/outputs.hpp"
#include "mcgpuconstants.hpp"

#include <stdio.h>



bool cudaCheck(const char* msg, cudaError_t errorCode){
	if (cudaSuccess != errorCode) {
		std::cout << (msg) << " " << cudaGetErrorString(errorCode) << std::endl;
		return false;
	}
	return true;
}


__device__ float computeSphere(int i, int j, int k, int gridSize, float value) {
	float x = (2 * i / float(gridSize)) - 1.0;
	float y = (2 * j / float(gridSize)) - 1.0;
	float z = (2 * k / float(gridSize)) - 1.0;
	return x * x + y * y + z * z - 0.5;


}

__device__ float fresnel(int i, int j, int k, int gridSize, float value) {
	float x = (2.0 * float(i) / float(gridSize)) - 1.0;
	float y = (2.0 * float(j) / float(gridSize)) - 1.0;
	float z = (2.0 * float(k) / float(gridSize)) - 1.0;
	return cos(value * (x*x + y*y + z*z));
}

__device__ float computeBallons(int i, int j, int k, int gridSize, float value) {
	float x = (2.0 * float(i) / float(gridSize)) - 1.0;
	float y = (2.0 * float(j) / float(gridSize)) - 1.0;
	float z = (2.0 * float(k) / float(gridSize)) - 1.0;

	const float schtroumf = value;
	return -(x*x + y*y + z*z) + cos(x * schtroumf) * cos(schtroumf * y) * cos(schtroumf * z) + 0.215;
}

/**
*	Fill grid with the isosurface function
*   grid is a grid[gridSize][gridSize][gridSize]
*/
__global__ void computeFonction(float *grid, int gridSize, float value)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < gridSize * gridSize * gridSize) {
		int ix = index / (gridSize * gridSize);
		int iy = (index / gridSize) % gridSize;
		int iz = index % gridSize;
		grid[index] = computeBallons(ix, iy, iz, gridSize, value);
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
	int indexVoxel = 0;
	int baseSize = gridSize - 1;
	if (ix < baseSize && iy < baseSize && iz < baseSize) {
		indexVoxel |= (grid[index] > 0) * 1;
		indexVoxel |= (grid[index + gridSize * gridSize] > 0) * 2;
		indexVoxel |= (grid[index + gridSize * gridSize + gridSize] > 0) * 4;
		indexVoxel |= (grid[index + gridSize] > 0) * 8;
		indexVoxel |= (grid[index + 1] > 0) * 16;
		indexVoxel |= (grid[index + gridSize * gridSize + 1] > 0) * 32;
		indexVoxel |= (grid[index + gridSize * gridSize + gridSize + 1] > 0) * 64;
		indexVoxel |= (grid[index + 1 + gridSize] > 0) * 128;
		oBasePyramid[ix * baseSize * baseSize + iy * baseSize + iz] = triTableSizes[indexVoxel];
	}
}

/**
* Build a pyramid level from the previous one
*/
__global__ void buildPyramidStep(int *oBasePyramid, int previousOffset, int currentOffset, int currentSize) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < currentSize) {
		oBasePyramid[currentOffset + index] = oBasePyramid[previousOffset + 2 * index];
		oBasePyramid[currentOffset + index] += oBasePyramid[previousOffset + 2 * index + 1];
	}
}

/**
*	Sum the values of each level of the pyramid to get the final offset of each voxel
*/
__global__ void buildPyramidResult(int *pyramid, int size, int *oPyramidResult) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < size) {
		int offset = 0;
		int currentIndex = index;
		oPyramidResult[index] = 0;
		while (size != 0) {
			//oPyramidResult[index] += pyramid[offset + currentIndex];
			if (currentIndex % 2 == 1) {
				oPyramidResult[index] += pyramid[offset + currentIndex - 1];
			}
			offset += size;
			size /= 2;
			currentIndex /= 2;
		}
	}
}

#define INTERPOL_VERTEX(EDGE_VALUE,	VERTEX, POSCELL, INDEX, P1, P2) \
if ((EDGE_VALUE) & (1 << (INDEX))) {\
	VERTEX[3 * (INDEX)] = POSCELL[0] + pointOffset[3 * ((P1))] - values[(P1)] * (pointOffset[3 * (P2)] - pointOffset[3 * (P1)]) / (values[(P2)] - values[(P1)]);\
    VERTEX[3 * (INDEX) + 1] = POSCELL[1] + pointOffset[3 * (P1) + 1] - values[(P1)] * (pointOffset[3 * (P2) + 1] - pointOffset[3 * (P1) + 1]) / (values[(P2)] - values[(P1)]);\
	VERTEX[3 * (INDEX) + 2] = POSCELL[2] + pointOffset[3 * (P1) + 2] - values[(P1)] * (pointOffset[3 * (P2) + 2] - pointOffset[3 * (P1) + 2]) / (values[(P2)] - values[(P1)]);\
}

/**
* @param output : output to write
* @param offset : offset in output
* @param posCell : the position of the current cellul in space coordinates
* @param values : the 8  values of the grid around the cell
* @param pointOffset, p1, p2
* @return the new offset in output
*/
__device__ void interpolateAndWrite(float *output, int offset, const float *posCell, const float *values,
	const float *pointOffset, int p1, int p2)
{
	output[offset] = posCell[0] + pointOffset[3 * p1] - values[p1] * (pointOffset[3 * p2] - pointOffset[3 * p1]) / (values[p2] - values[p1]);
	output[offset + 1] = posCell[1] + pointOffset[3 * p1 + 1] - values[p1] * (pointOffset[3 * p2 + 1] - pointOffset[3 * p1 + 1]) / (values[p2] - values[p1]);
	output[offset + 2] = posCell[2] + pointOffset[3 * p1 + 2] - values[p1] * (pointOffset[3 * p2 + 2] - pointOffset[3 * p1 + 2]) / (values[p2] - values[p1]);
}


__global__ void fillTrianglesGPU(const int *pyramidResult, const float *grid, int pyramidSize, int gridSize,
	const int *edgeTable, int *triTable, const float *pointOffset,
	float *outputTriangles) {

	int indexGrid = threadIdx.x + blockIdx.x * blockDim.x;
	int ix = indexGrid / (gridSize * gridSize);
	int iy = (indexGrid / gridSize) % gridSize;
	int iz = indexGrid % gridSize;
	int indexVoxel = 0;
	int baseSize = gridSize - 1;
	float values[8];
	if (ix < baseSize && iy < baseSize && iz < baseSize) {
		int indexPyramid = ix * baseSize * baseSize + iy * baseSize + iz;
		values[0] = grid[indexGrid];
		values[1] = grid[indexGrid + gridSize * gridSize];
		values[2] = grid[indexGrid + gridSize * gridSize + gridSize];
		values[3] = grid[indexGrid + gridSize];
		values[4] = grid[indexGrid + 1];
		values[5] = grid[indexGrid + gridSize * gridSize + 1];
		values[6] = grid[indexGrid + gridSize * gridSize + gridSize + 1];
		values[7] = grid[indexGrid + 1 + gridSize];

		indexVoxel |= (values[0] > 0) * 1;
		indexVoxel |= (values[1] > 0) * 2;
		indexVoxel |= (values[2] > 0) * 4;
		indexVoxel |= (values[3] > 0) * 8;
		indexVoxel |= (values[4] > 0) * 16;
		indexVoxel |= (values[5] > 0) * 32;
		indexVoxel |= (values[6] > 0) * 64;
		indexVoxel |= (values[7] > 0) * 128;
		int edgeValue = edgeTable[indexVoxel];
		if (!edgeValue) { // no triangle
			return;
		}
		float posCell[3] = { float(ix) / float(gridSize), float(iy) / float(gridSize), float(iz) / float(gridSize) };
		int offsetTriangles = pyramidResult[indexPyramid] * 3;
		for (int i = 0; triTable[indexVoxel * 16 + i] != -1; ++i) {
			int tri = triTable[indexVoxel * 16 + i];
			int p1 = tri % 8;
			int p2 = (tri < 8) ? ((tri + 1) % 4 + (tri / 4) * 4) : (tri - 4);
			interpolateAndWrite(outputTriangles, offsetTriangles, posCell, values, pointOffset, p1, p2);
			offsetTriangles += 3;
		}
	}

}


MarchingCubes::MarchingCubes(int gridSize) :
PYRAMID_GRID_SIZE(gridSize),
PYRAMID_TOTAL_GRID_SIZE(PYRAMID_GRID_SIZE *PYRAMID_GRID_SIZE *PYRAMID_GRID_SIZE),
FCT_GRID_SIZE(gridSize + 1),
FCT_TOTAL_GRID_SIZE(FCT_GRID_SIZE * FCT_GRID_SIZE * FCT_GRID_SIZE),
cudaBlockSize(512),
cudaGridSize(FCT_TOTAL_GRID_SIZE / 512),
N(1 << (int)ceil(log2((double)PYRAMID_TOTAL_GRID_SIZE)))
{
	// declarations
	functionValuesGPU = 0;
	triTableSizesGPU = 0;
	histoPyramidBaseGPU = 0;
	histoPyramidResultGPU = 0;
	pointsOffsetGPU = 0;
	edgeTableGPU = 0;
	triTableGPU = 0;
	outputTrianglesGPU = 0;
	maxPointsNumber = 0;

	const float UNITOFFSET = 1.0 / float(FCT_GRID_SIZE);
	float pointOffsetTemp[24] = { 0.0, 0.0, 0.0,
		UNITOFFSET, 0.0, 0.0,
		UNITOFFSET, UNITOFFSET, 0.0,
		0.0, UNITOFFSET, 0.0,
		0.0, 0.0, UNITOFFSET,
		UNITOFFSET, 0.0, UNITOFFSET,
		UNITOFFSET, UNITOFFSET, UNITOFFSET,
		0.0, UNITOFFSET, UNITOFFSET };
	memcpy(pointOffset, pointOffsetTemp, 24 * sizeof(float));

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << std::endl;
		return;
	}


	// allocate
	cudaCheck("functionValuesGPU allocation failed", cudaMalloc((void**)&functionValuesGPU, FCT_TOTAL_GRID_SIZE * sizeof(float)));
	cudaCheck("triTableSizesGPU allocation failed", cudaMalloc((void**)&triTableSizesGPU, 256 * sizeof(int)));
	cudaCheck("pointsOffsetGPU allocation failed", cudaMalloc((void**)&pointsOffsetGPU, 24 * sizeof(float)));
	cudaCheck("edgeTableGPU allocation failed", cudaMalloc((void**)&edgeTableGPU, 256 * sizeof(int)));
	cudaCheck("triTableGPU allocation failed", cudaMalloc((void**)&triTableGPU, 256 * 16 * sizeof(int)));
	cudaCheck("histoPyramidBaseGPU allocation failed", cudaMalloc((void**)&histoPyramidBaseGPU, 2 * N * sizeof(int)));
	cudaCheck("histoPyramidResultGPU allocation failed", cudaMalloc((void**)&histoPyramidResultGPU, N * sizeof(int)));

	// fill constant values
	// could be parallel with previous things
	// there is a faster memory to use (todo, mbe in the alloc step)
	cudaCheck("cannot fill tritablesizes", cudaMemcpy(triTableSizesGPU, triTableSizes, 256 * sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck("cannot fill pointsOffsetGPU", cudaMemcpy(pointsOffsetGPU, pointOffset, 24 * sizeof(float), cudaMemcpyHostToDevice));
	cudaCheck("cannot fill edgeTableGPU", cudaMemcpy(edgeTableGPU, edgeTable, 256 * sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck("cannot fill triTable", cudaMemcpy(triTableGPU, triTable, 256 * 16 * sizeof(int), cudaMemcpyHostToDevice));
}

MarchingCubes::~MarchingCubes(){
	cudaFree(functionValuesGPU);
	cudaFree(triTableSizesGPU);
	cudaFree(pointsOffsetGPU);
	cudaFree(edgeTableGPU);
	cudaFree(triTableGPU);
	cudaFree(histoPyramidBaseGPU);
	cudaFree(histoPyramidResultGPU);
	cudaDeviceReset();
}

void MarchingCubes::computeTriangles(std::vector<float> &triangles, float value) {
	triangles.clear();
	// fill values grid
	computeFonction << <cudaGridSize, cudaBlockSize >> >(functionValuesGPU, FCT_GRID_SIZE, value);
	cudaCheck("computeFonction failed", cudaGetLastError());


	// compute base pyramid
	buildBasePyramid << <cudaGridSize, cudaBlockSize >> >(triTableSizesGPU, functionValuesGPU, FCT_GRID_SIZE, histoPyramidBaseGPU);
	cudaCheck("buildBasePyramid failed", cudaGetLastError());


	// compute all pyramids levels
	int previousOffset = 0;
	int pointsNumber = 0;
	int size = N;
	while (size > 1) {
		int newOffset = previousOffset + size;
		size /= 2;
		buildPyramidStep << <cudaGridSize, cudaBlockSize >> >(histoPyramidBaseGPU, 
			previousOffset, newOffset, size);
		previousOffset = newOffset;
	}
	cudaCheck("buildBasePyramid failed", cudaGetLastError());


		// get final offsets
	buildPyramidResult << <cudaGridSize, cudaBlockSize >> >(histoPyramidBaseGPU, 
		N, histoPyramidResultGPU);
	cudaCheck("buildBasePyramid failed", cudaGetLastError());


	cudaCheck("get triangles number failed", cudaMemcpy(&pointsNumber,
		&(histoPyramidResultGPU[PYRAMID_TOTAL_GRID_SIZE - 1]), 1 * sizeof(int), cudaMemcpyDeviceToHost));
	std::cout << "points number : " << pointsNumber << std::endl;
	if (!pointsNumber) {
		return;
	}
	if (pointsNumber > maxPointsNumber) {
		maxPointsNumber = maxPointsNumber;
		cudaFree(outputTrianglesGPU);
		cudaMalloc((void**)&outputTrianglesGPU, pointsNumber * 3 * sizeof(float));
		cudaCheck("histoPyramidResultGPU allocation failed", cudaGetLastError());
	}



	fillTrianglesGPU << <cudaGridSize, cudaBlockSize >> >(histoPyramidResultGPU, 
		functionValuesGPU, PYRAMID_GRID_SIZE, FCT_GRID_SIZE,
		edgeTableGPU, triTableGPU, pointsOffsetGPU, outputTrianglesGPU);
	triangles = std::vector<float>(pointsNumber * 3);
	cudaCheck("copy triangles from gpu", cudaMemcpy(&(triangles[0]), outputTrianglesGPU,
		pointsNumber * 3 * sizeof(float), cudaMemcpyDeviceToHost));

	cudaCheck("fillTrianglesGPU failed", cudaGetLastError());
}

