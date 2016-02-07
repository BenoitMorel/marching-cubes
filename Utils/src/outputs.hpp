#ifndef _OUTPUT_H_
#define _OUTPUT_H_
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <float.h>
#include <algorithm>

void writePGMToFile(std::string fileName, int width, int height, int *data);


void writePGMToFileInt(std::string fileName, int width, int height, int *data) {
	int min = 999999999;
	int max = -99999999;
	for (int i = 0; i < width * height; ++i) {
		min = std::min(data[i], min);
		max = std::max(data[i], max);
	}
	if (min == max) {
		max = min + 1;
	}
	std::vector<int> newData(width * height);
	for (int i = 0; i < width * height; ++i) {
		int plop = (((data[i] - min) * 255) / (max - min));
		newData[i] = plop;
	}
	writePGMToFile(fileName, width, height, &newData[0]);
}

void writePGMToFileChar(std::string fileName, int width, int height, char *data) {
	char min = 127;
	char max = -127;
	for (int i = 0; i < width * height; ++i) {
		min = std::min(data[i], min);
		max = std::max(data[i], max);
	}
	if (min == max) {
		max = min + 1;
	}
	std::vector<int> newData(width * height);
	for (int i = 0; i < width * height; ++i) {
		int plop = (((data[i] - min) * 255) / (max - min));
		newData[i] = plop;
	}
	writePGMToFile(fileName, width, height, &newData[0]);
}

void writePGMToFileFloat(std::string fileName, int width, int height, float *data) {
	float min = FLT_MAX;
	float max = FLT_MIN;
	for (int i = 0; i < width * height; ++i) {
		min = std::min(data[i], min);
		max = std::max(data[i], max);
	}
	std::vector<int> newData(width * height);
	for (int i = 0; i < width * height; ++i) {
		int plop = (((data[i] - min) * 255.0) / (max - min) + 0.0001);
		newData[i] = plop;
	}
	writePGMToFile(fileName, width, height, &newData[0]);
}

void writePGMToFile(std::string fileName, int width, int height, int *data)
{
	std::ofstream os(fileName);
	os << "P2" << " "
		<< width << " "
		<< height << " "
		<< 255 << " "
		;
	for (int i = 0; i < width * height; ++i) {
		os << (int)data[i] << " ";
	}
}
/*
writePyramid(std::string fileName, int size, int *pyramid) {
	int 
}
*/
void writeData(std::string fileName, int* values, int size) {
	std::ofstream os(fileName);
	for (int i = 0; i < size; ++i) {
		os << values[i] << " ";
	}
}

#endif