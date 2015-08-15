#ifndef GG_CAMERA_H
#define GG_CAMERA_H

#include <iostream>
#include <algorithm>

#include <glfw3.h>
#include <glm/glm.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>


// Callback
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);


class Camera {
	float horizontalAngle;
	float verticalAngle;
	float initialFov;
	glm::mat4 projectionMatrix;
	glm::mat4 viewMatrix;
	glm::mat4 rotation;
	glm::vec3 translation;
	const glm::mat4 initViewMatrix;

	double previousXLeft;
	double previousYLeft;
	bool leftClickCount;

	double previousXRight;
	double previousYRight;
	bool rightClickCount;
public:
	static bool leftButtonPressed;
	static bool rightButtonPressed;
	static int wheelRotation;

public: 
	Camera();
	glm::vec3 getSphereVec(double x, double y, double r);

	void update(GLFWwindow* window, float deltaTime);
	void updateLeft(int width, int height, double mouseX, double mouseY);
	void updateRight(int width, int height, double mouseX, double mouseY);

	glm::mat4 &getProjection() {
		return projectionMatrix;
	}

	glm::mat4 &getView() {
		return viewMatrix;
	}

};

#endif