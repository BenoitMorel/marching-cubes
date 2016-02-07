#ifndef _GG_VIEWER_
#define _GG_VIEWER_

#include <GL/glew.h>

#include <glfw3.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <glm/glm.hpp>
using namespace glm;
#include "shader.hpp"
#include "texture.hpp"
#include "Constants.hpp"
#include "Camera.hpp"
#include "Updatable.hpp"
#include <vector>
#include <time.h>
#include <iostream>


class Viewer {
public:
	Viewer();
	~Viewer();
	bool init();
	bool loop();

	/**
	*	Set triangles in the VBO. (without normals or anything)
	*/
	void setTriangles(float *triangles, int trianglesNumber);

	void addUpdatable(Updatable *updatable);
	
private:
	bool ok;
	GLFWwindow* window;
	GLuint vaoID;
	GLuint programID;
	Camera camera;
	GLuint matrixID;
	double lastTime;

	int _trianglesNumber;
	GLuint vboID;
	int vboIDSize;
	std::vector<Updatable *> toUpdate;

private:
	void resize(int size);
};

#endif