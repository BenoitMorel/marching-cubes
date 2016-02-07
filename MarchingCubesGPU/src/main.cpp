#include "../../OpenglViewer/src/Viewer.hpp"
#include "kernel.cuh"

#include <fstream>


/*
void computeSizeTable() {
	std::ofstream stream("c:\\temp\\table.txt");
	stream << "float triTableSizes = {";
	for (int i = 0; i < 256; ++i) {
		int j = 0;
		for (; triTable[i][j] != -1; ++j) {}
		stream << j << ",";
	}
	stream << "};";
	stream.close();
}
int f(int tri) {
	return (tri < 8) ? ((tri + 1) % 4 + (tri / 4) * 4) : (tri - 4);
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

int main() {
	std::vector<float> points;
	computeTriangles(points);


	points.push_back(0);
	points.push_back(0);
	points.push_back(0);

	points.push_back(0);
	points.push_back(1);
	points.push_back(0);

	points.push_back(0);
	points.push_back(0);
	points.push_back(1);
	
	Viewer viewer;
	viewer.setTriangles(&points[0], points.size() / 9);
	while (viewer.loop()) {}
	
	std::cout << "end of main" << std::endl;
	
	//int plop; std::cin >> plop;
}*/