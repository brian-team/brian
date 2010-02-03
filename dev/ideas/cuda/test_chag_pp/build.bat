call "C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin\vcvars32.bat"
nvcc -I"D:\Programming" -I"C:\Python26\include" -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\bin" "-IC:\Program Files (x86)\Microsoft Visual Studio 9.0\VC\include" -arch compute_11 testchagpp.cu -shared -o testchagpp.dll
