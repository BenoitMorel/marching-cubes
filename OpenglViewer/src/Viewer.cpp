#include "Viewer.hpp"



Viewer::Viewer()
{
	ok = init();
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

	// texture 
	int picWidth = 256;
	int picHeight = 256;

	GLuint Texture = GG::loadBMP_custom("..\\ressources\\textures\\uvtemplate.bmp");

	//VBO
	glGenBuffers(1, &vboID);
	glBindBuffer(GL_ARRAY_BUFFER, vboID);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	glGenBuffers(1, &colorVboID);
	glBindBuffer(GL_ARRAY_BUFFER, colorVboID);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_uv_buffer_data) * sizeof(GLfloat), g_uv_buffer_data, GL_STATIC_DRAW);

	// shaders
	programID = GG::LoadShaders("..\\shaders\\SimpleVertexShader.vertexshader", "..\\shaders\\SimpleFragmentShader.fragmentshader");

	matrixID = glGetUniformLocation(programID, "MVP");
	lastTime = glfwGetTime() - 1;

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
	deltaTime = lastTime - currentTime;
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

	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, colorVboID);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glUseProgram(programID);

	glUniformMatrix4fv(matrixID, 1, GL_FALSE, &(MVP[0][0]));

	glDrawArrays(GL_TRIANGLES, 0, 12 * 3);

	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(0);


	glfwSwapBuffers(window);
	glfwPollEvents();

	return glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
		glfwWindowShouldClose(window) == 0;
}