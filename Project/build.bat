mkdir build
cd build
cmake -G"Visual Studio 15 2017 Win64" ..
set VCINSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio 15.0\VC
cmake --build . --config Release
