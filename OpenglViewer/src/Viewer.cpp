#include "Viewer.hpp"



Viewer::Viewer()
{
	ok = init();
	_trianglesNumber = 0;
	glGenBuffers(1, &vboID);
}

void Viewer::setTriangles(float *triangles, int trianglesNumber)
{
	static int i = 0;
	if (i == 0) {
		_trianglesNumber = trianglesNumber;
		glBindBuffer(GL_ARRAY_BUFFER, vboID);
		glBufferData(GL_ARRAY_BUFFER, trianglesNumber * 9 * sizeof(float), triangles, GL_DYNAMIC_DRAW);
		i++;
	}
	else {
		_trianglesNumber = trianglesNumber;
		glBindBuffer(GL_ARRAY_BUFFER, vboID);
		if (trianglesNumber) {
			glBufferSubData(GL_ARRAY_BUFFER, 0, trianglesNumber * 9 * sizeof(float), triangles);
			std::cout << "update buffer " << trianglesNumber << std::endl;
		}
	}


}


bool Viewer::init()
{
	// Initialise GLFW
	if (!glfwInit())
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
		return false;
	}
	
	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(1024, 768, "Playground", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return false;
	}
	glfwMakeContextCurrent(window);
	glewExperimental = true; 

	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return false;
	}

	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetScrollCallback(window, scroll_callback);

	glGenVertexArrays(1, &vaoID);
	glBindVertexArray(vaoID);

	// shaders
	programID = GG::LoadShaders("..\\..\\OpenglViewer\\shaders\\SimpleVertexShader.vertexshader", "..\\..\\OpenglViewer\\shaders\\SimpleFragmentShader.fragmentshader");

	matrixID = glGetUniformLocation(programID, "MVP");
	lastTime = glfwGetTime() - 0.000000001;

	return true;
}

bool Viewer::loop()
{
	if (!ok) {
		return false;
	}
	// update
	int width, height;
	float deltaTime;
	double currentTime = glfwGetTime();
	deltaTime = currentTime - lastTime;

	for (int i = 0; i < toUpdate.size(); ++i) {
		toUpdate[i]->update(deltaTime);
	}

	lastTime = currentTime;
	camera.update(window, deltaTime);
	glm::mat4 MVP = camera.getProjection() * camera.getView();


	// render
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vboID);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glUseProgram(programID);

	glUniformMatrix4fv(matrixID, 1, GL_FALSE, &(MVP[0][0]));

	glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
	if (_trianglesNumber) {
		glDrawArrays(GL_TRIANGLES, 0, _trianglesNumber * 3);
		std::cout << "draw " << _trianglesNumber << std::endl;
	}

	glDisableVertexAttribArray(0);


	glfwSwapBuffers(window);
	glfwPollEvents();

	return glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
		glfwWindowShouldClose(window) == 0;
}

void Viewer::addUpdatable(Updatable *updatable) {
	toUpdate.push_back(updatable);
}