c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) src/cpp_modules.cpp -o feature_extractor_cpp$(python3-config --extension-suffix)
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) src/cpp_modules.cpp -o segment_utils_cpp$(python3-config --extension-suffix)
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) src/cpp_modules.cpp -o quantization_utils_cpp$(python3-config --extension-suffix)
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) src/cpp_modules.cpp -o dataset_utils_cpp$(python3-config --extension-suffix)
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) src/cpp_modules.cpp -o contour_utils_cpp$(python3-config --extension-suffix)

