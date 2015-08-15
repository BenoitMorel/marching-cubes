
#include <stdio.h>
#include <stdlib.h>

#include "Viewer.hpp"


int main( void )
{
	Viewer viewer;
	while (viewer.loop()) {}
	return 0;
}

