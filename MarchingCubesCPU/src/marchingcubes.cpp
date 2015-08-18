
#include <iostream>
#include <fstream>
#include "../../OpenglViewer/src/Viewer.hpp"
#include "marchingcubesconstants.hpp"

#define GETINDEX(macroi, macroj,macrok) (((macroi) * GRID_SIZE * GRID_SIZE) + ((macroj) * GRID_SIZE) + (macrok))
#define GETK(macroindex) ((macroindex) % GRID_SIZE)
#define GETJ(macroindex) (((macroindex)/GRID_SIZE) % GRID_SIZE)
#define GETI(macroindex) ((macroindex) / (GRID_SIZE * GRID_SIZE))

/*
*	isosurfaces from http ://www.z-way.org/script-and-gizmo/houdini/isosurface
*/

static int test = 0;

float fresnel(float x, float y, float z) {
	return cos(10 * (x*x + y*y + z*z));
}

float ballons(float x, float y, float z) {

	x *= 2.0;
	y *= 2.0;
	z *= 2.0;
	x -= 1;
	y -= 1;
	z -= 1;
	const int plop = 1000 + test;
	return -(x*x + y*y + z*z) + cos(x * plop) * cos(plop * y) * cos(plop * z) + 0.215;
}

float myCrazyFunction(float x, float y, float z)
{
	// describes a sphere of radius 1
	x *= 2.0;
	y *= 2.0;
	z *= 2.0;
	x -= 1;
	y -= 1;
	z -= 1;
	return x * x * x + sqrt(fabs(y)) * x + z * z - 0.9f;
}

#define INTERPOL_VERTEX(EDGE_VALUE,	VERTEX, POSCELL, INDEX, P1, P2) \
if ((EDGE_VALUE) & (1 << (INDEX))) {\
	VERTEX[3 * (INDEX)] = POSCELL[0] + pointOffset[3 * ((P1))] - values[(P1)] * (pointOffset[3 * (P2)] - pointOffset[3 * (P1)]) / (values[(P2)] - values[(P1)]);\
    VERTEX[3 * (INDEX) + 1] = POSCELL[1] + pointOffset[3 * (P1) + 1] - values[(P1)] * (pointOffset[3 * (P2) + 1] - pointOffset[3 * (P1) + 1]) / (values[(P2)] - values[(P1)]);\
	VERTEX[3 * (INDEX) + 2] = POSCELL[2] + pointOffset[3 * (P1) + 2] - values[(P1)] * (pointOffset[3 * (P2) + 2] - pointOffset[3 * (P1) + 2]) / (values[(P2)] - values[(P1)]);\
}

void getTrianglesFrom(int i, int j, int k, float *grid, std::vector<float> &output) {

	int index = 0; // each bit == 1 if corresponding vertex > 0
	float values[8];
	values[0] = grid[GETINDEX(i, j, k)];
	values[1] = grid[GETINDEX(i + 1, j, k)];
	values[2] = grid[GETINDEX(i + 1, j + 1, k)];
	values[3] = grid[GETINDEX(i, j + 1, k)];
	values[4] = grid[GETINDEX(i, j, k + 1)];
	values[5] = grid[GETINDEX(i + 1, j, k + 1)];
	values[6] = grid[GETINDEX(i + 1, j + 1, k + 1)];
	values[7] = grid[GETINDEX(i, j + 1, k + 1)];
	index |= (values[0] > 0) * 1;
	index |= (values[1] > 0) * 2;
	index |= (values[2] > 0) * 4;
	index |= (values[3] > 0) * 8;
	index |= (values[4] > 0) * 16;
	index |= (values[5] > 0) * 32;
	index |= (values[6] > 0) * 64;
	index |= (values[7] > 0) * 128;
	int edgeValue = edgeTable[index];
	if (!edgeValue) { // no triangle
		return;
	}
	float vertices[36];
	for (int i = 0; i < 36; ++i) {
		vertices[i] = 666666;
	}
	float posCell[3] = { float(i) / float(GRID_SIZE), float(j) / float(GRID_SIZE), float(k) / float(GRID_SIZE) };
	INTERPOL_VERTEX(edgeValue, vertices, posCell, 0, 0, 1);
	INTERPOL_VERTEX(edgeValue, vertices, posCell, 1, 1, 2);
	INTERPOL_VERTEX(edgeValue, vertices, posCell, 2, 2, 3);
	INTERPOL_VERTEX(edgeValue, vertices, posCell, 3, 3, 0);
	INTERPOL_VERTEX(edgeValue, vertices, posCell, 4, 4, 5);
	INTERPOL_VERTEX(edgeValue, vertices, posCell, 5, 5, 6);
	INTERPOL_VERTEX(edgeValue, vertices, posCell, 6, 6, 7);
	INTERPOL_VERTEX(edgeValue, vertices, posCell, 7, 7, 4);
	INTERPOL_VERTEX(edgeValue, vertices, posCell, 8, 0, 4);
	INTERPOL_VERTEX(edgeValue, vertices, posCell, 9, 1, 5);
	INTERPOL_VERTEX(edgeValue, vertices, posCell, 10, 2, 6);
	INTERPOL_VERTEX(edgeValue, vertices, posCell, 11, 3, 7);
	for (int ii = 0; triTable[index][ ii] != -1; ii += 3) {
		output.push_back(vertices[    3 * triTable[index][ ii]]);
		output.push_back(vertices[1 + 3 * triTable[index][ ii]]);
		output.push_back(vertices[2 + 3 * triTable[index][ii]]);

		output.push_back(vertices[    3 * triTable[index][ ii + 1]]);
		output.push_back(vertices[1 + 3 * triTable[index][ii + 1]]);
		output.push_back(vertices[2 + 3 * triTable[index][ii + 1]]);

		output.push_back(vertices[    3 * triTable[index][ii + 2]]);
		output.push_back(vertices[1 + 3 * triTable[index][ii + 2]]);
		output.push_back(vertices[2 + 3 * triTable[index][ii + 2]]);
		
	}
}

void writeObj(char * fileName, float *data, int dataSize) {
	std::ofstream os(fileName);
	for (int i = 0; i < dataSize; i += 9) {
		int off = i;
		os << "v " << data[off] << " " << data[off + 1] << " " << data[off + 2] << std::endl;
		off += 3;
		os << "v " << data[off] << " " << data[off + 1] << " " << data[off + 2] << std::endl;
		off += 3;
		os << "v " << data[off] << " " << data[off + 1] << " " << data[off + 2] << std::endl;
	}
	for (int i = 0; i < dataSize; i += 9) {
		os << "f " << i / 3 + 1 << " " << i / 3 + 2 << " " << i / 3 + 3 << std::endl;
	}

}

void compute(std::vector<float> &out) {
	std::vector<float> grid(GRID_SIZE * GRID_SIZE * GRID_SIZE);
	for (int i = 0; i < GRID_SIZE * GRID_SIZE * GRID_SIZE; ++i) {
		grid[i] = ballons(float(GETI(i)) / float(GRID_SIZE), float(GETJ(i)) / float(GRID_SIZE), float(GETK(i) / float(GRID_SIZE)));
	}
	for (int i = 0; i < GRID_SIZE - 1; ++i) {
		for (int j = 0; j < GRID_SIZE - 1; ++j) {
			for (int k = 0; k < GRID_SIZE - 1; ++k) {
				getTrianglesFrom(i, j, k, &grid[0], out);
			}
		}
	}
	std::cout << out.size() << std::endl;
}

int main(void)
{

	
	std::vector<float> outputPoints;
	compute(outputPoints);
    Viewer viewer;
	viewer.setTriangles(&outputPoints[0], outputPoints.size() / 9);
	//double lastTime = glfwGetTime();
	while (viewer.loop()) {
		/*if (glfwGetTime() - lastTime > 2.0) {
			test += 200;
			outputPoints.clear();
			compute(outputPoints);
			viewer.setTriangles(&outputPoints[0], outputPoints.size() / 9);
			lastTime = glfwGetTime();
		}*/
	}
	return 1;
}
