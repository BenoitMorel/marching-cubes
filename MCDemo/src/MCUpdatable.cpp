#include "MCUpdatable.hpp"
#include <iostream>

#include "../../MarchingCubesGPU/src/kernel.cuh"


MCUpdatable::MCUpdatable(Viewer &viewer):
_viewer(viewer),
_marchingCubes(128),
_currentValue(0.0),
_fps()
{

}

MCUpdatable::~MCUpdatable() {

}


void MCUpdatable::update(float elapsed) {
	_currentValue += elapsed;
	_fps.update(elapsed);
	//_fps.print();
	static float plop = 5;
	if (_currentValue > 0.1) {
		_triangles.clear();
		_marchingCubes.computeTriangles(_triangles, plop);
		plop += 0.1;
		std::cout << "plop " << plop << std::endl;
		_viewer.setTriangles(_triangles.size() ? &(_triangles[0]) : 0, _triangles.size() / 9);
		_currentValue = 0.0;
		if (plop > 10) {
			plop = 5;
		}
	}
}