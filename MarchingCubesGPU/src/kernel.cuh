#ifndef _KEEERNEL_
#define _KEEERNEL_
#include <vector>

#include "cuda_runtime.h"


class MarchingCubes {
public:
	MarchingCubes(int gridSize);
	~MarchingCubes();
	void computeTriangles(std::vector<float> &triangles, float value);

private:
	const int PYRAMID_GRID_SIZE;
	const int PYRAMID_TOTAL_GRID_SIZE;
	const int FCT_GRID_SIZE;
	const int FCT_TOTAL_GRID_SIZE;
	const int N; // allocated size for the base of the pyramid
	const dim3 cudaBlockSize;
	const dim3 cudaGridSize;

	float *functionValuesGPU;
	int *triTableSizesGPU;
	int *histoPyramidBaseGPU;
	int *histoPyramidResultGPU;
	float *pointsOffsetGPU;
	int *edgeTableGPU;
	int *triTableGPU;
	float *outputTrianglesGPU;

	int maxPointsNumber;
	float pointOffset[24];


};

#endif