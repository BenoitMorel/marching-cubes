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
}*/

int main() {
	std::vector<float> points;
	computeTriangles(points);
	/*
	Viewer viewer;
	viewer.setTriangles(&points[0], points.size() / 9);
	while (viewer.loop()) {}
	*/
	
	std::cout << "end of main" << std::endl;

}