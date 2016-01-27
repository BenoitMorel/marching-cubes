#ifndef _KEEERNEL_
#define _KEEERNEL_
#include <vector>


const int FCT_GRID_SIZE = 65;
const int FCT_TOTAL_GRID_SIZE = FCT_GRID_SIZE * FCT_GRID_SIZE * FCT_GRID_SIZE;
const int PYRAMID_GRID_SIZE = FCT_GRID_SIZE - 1;
const int PYRAMID_TOTAL_GRID_SIZE = PYRAMID_GRID_SIZE * PYRAMID_GRID_SIZE * PYRAMID_GRID_SIZE;

void computeTriangles(std::vector<float> &triangles);

#endif