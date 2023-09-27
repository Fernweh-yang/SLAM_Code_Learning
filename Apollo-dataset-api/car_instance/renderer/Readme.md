### Instruction for renderer
We modify the renderer based on the render crom from the code provided by [displet](http://www.cvlibs.net/projects/displets/)

Dependency: `python-tk libeigen3-dev libglfw-dev libgles2-mesa-dev libglew-dev libboost-all-dev`

Two things are modified: (1) we give the renderer a python wrapper (2) we provide an egl context so the the render can be performed off-screen.

Tested with Ubuntu 14.04 and Python 2.7, nvidia diver 375.26. For other versions, we haven't tested it. 

