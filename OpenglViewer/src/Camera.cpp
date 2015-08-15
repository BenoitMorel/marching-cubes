#include "Camera.hpp"


bool Camera::leftButtonPressed = false;
bool Camera::rightButtonPressed = false;
int Camera::wheelRotation = 0;

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		Camera::leftButtonPressed = (action == GLFW_PRESS);
	}
	if (button == GLFW_MOUSE_BUTTON_RIGHT) {
		Camera::rightButtonPressed = (action == GLFW_PRESS);
	}
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	Camera::wheelRotation = std::min(45.0, std::max(-45.0, Camera::wheelRotation + yoffset * 2));
	
	std::cout << yoffset << std::endl;
}



Camera::Camera() : initViewMatrix(glm::lookAt(glm::vec3(0, 0, 5), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0)))
{
	rotation = glm::mat4(1.0);
	viewMatrix = initViewMatrix;
	horizontalAngle = 3.14;
	verticalAngle = 0;
	initialFov = 45.0;
	leftClickCount = false;
	rightClickCount = false;
}

glm::vec3 Camera::getSphereVec(double x, double y, double r)
{
	glm::vec3 res(x, y, 0);
	double xxyy = x*x + y*y;
	if (xxyy < r *r / 2.0) {
		res.z = sqrt(r * r - (xxyy));
	}
	else {
		res.z = r * r / (2 * sqrt(xxyy));
	}
	res = glm::normalize(res);
	return res;
}


void Camera::updateRight(int width, int height, double mouseX, double mouseY) {
	if (!rightClickCount) {
		rightClickCount = true;
		previousXRight = mouseX;
		previousYRight = mouseY;
		return;
	}
	if (previousXRight == mouseX && previousYRight == mouseY) {
		return;
	}
	translation += glm::vec3(2 * (mouseX - previousXRight) / (double)width, 2 * (mouseY - previousYRight) / (double)height, 0.0);

	previousXRight = mouseX;
	previousYRight = mouseY;
}

void Camera::updateLeft(int width, int height, double mouseX, double mouseY) {
	if (!leftClickCount) {
		leftClickCount = true;
		previousXLeft = mouseX;
		previousYLeft = mouseY;
		return;
	}
	if (previousXLeft == mouseX && previousYLeft == mouseY) {
		return;
	}
	double r = std::min(width / 2, height / 2);
	glm::vec3 v1 = getSphereVec(previousXLeft, previousYLeft, r);
	glm::vec3 v2 = getSphereVec(mouseX, mouseY, r);
	glm::vec3 N = glm::normalize(glm::cross(v1, v2));
	double teta = acos(glm::dot(v1, v2)) * 180.0 / 3.1415926535;
	glm::mat4 rot = glm::rotate((float)teta, N);
	rotation = rot * rotation;
	previousXLeft = mouseX;
	previousYLeft = mouseY;
}

void Camera::update(GLFWwindow* window, float deltaTime) {

	int width, height;
	double mouseX, mouseY;
	glfwGetWindowSize(window, &width, &height);
	glfwGetCursorPos(window, &mouseX, &mouseY);
	projectionMatrix = glm::perspective(initialFov - wheelRotation, (float)4 / (float)3, 0.1f, 100.0f);

	mouseX -= width / 2;
	mouseY -= height / 2;
	mouseY = -mouseY;

	if (!leftButtonPressed) {
		leftClickCount = false;
	} else {
		updateLeft(width, height, mouseX, mouseY);
	}
	if (!rightButtonPressed) {
		rightClickCount = false;
	} else {
		updateRight(width, height, mouseX, mouseY);
	}
	viewMatrix = initViewMatrix * glm::translate(translation) * rotation;

	
}
