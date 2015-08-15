#ifndef MYSHADER_HPP
#define MYSHADER_HPP

#include <GL/glew.h>

namespace GG {
	GLuint LoadShaders(const char * vertex_file_path, const char * fragment_file_path);
}

#endif
