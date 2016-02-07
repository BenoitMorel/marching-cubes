#include "../../OpenglViewer/src/Viewer.hpp"
#include "../../MarchingCubesGPU/src/kernel.cuh"
#include "MCUpdatable.hpp"

int main() {
	Viewer viewer;
	MCUpdatable mcup(viewer);
	viewer.addUpdatable(&mcup);
	while (viewer.loop()) {}
	return 0;
}