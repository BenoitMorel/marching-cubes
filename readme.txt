includes :
$(SolutionDir)\..\ThirdParty\glew\include;$(SolutionDir)\..\ThirdParty\glm;$(SolutionDir)\..\ThirdParty\glfw\include\GLFW;$(VC_IncludePath);

libs:
$(SolutionDir)\..\ThirdParty\glew\lib;$(SolutionDir)\..\ThirdParty\glfw\lib;
opengl32.lib;glfw3.lib;glew32.lib;kernel32.lib
Ajouter les deps entre projets a

dyn libs:
PATH=%PATH%;$(SolutionDir)\..\ThirdParty\glew\bin

Supplémentaires :
$(SolutionDir)\..\GloupsSolution\Debug
OpenglViewer.lib