mkdir build
cd build
cmake -G"Visual Studio 15 2017 Win64" -DCMAKE_PREFIX_PATH="d:/glew-2.1.0" -DCMAKE_LIBRARY_PATH="d:glew-2.1.0/lib/Release/x64" ..
set VCINSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio 15.0\VC
cmake --build . --config Release
