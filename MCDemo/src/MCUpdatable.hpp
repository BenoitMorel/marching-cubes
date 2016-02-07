#ifndef _MCUPDATABLE_HPP_
#define _MCUPDATABLE_HPP_

#include "../../OpenglViewer/src/Updatable.hpp"
#include "../../Utils/src/FPS.hpp"
#include "../../OpenglViewer/src/Viewer.hpp"
#include "../../MarchingCubesGPU/src/kernel.cuh"
#include <vector>

class MCUpdatable : public Updatable {
public:
	MCUpdatable(Viewer &viewer);
	virtual ~MCUpdatable();
	virtual void update(float elasped);


private:
	Viewer &_viewer;
	MarchingCubes _marchingCubes;
	float _currentValue;
	FPS _fps;
	std::vector<float> _triangles;
};


#endif