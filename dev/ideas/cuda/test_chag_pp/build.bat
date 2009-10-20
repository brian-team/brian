del *.obj
del *.dll
call "C:\Program Files\Microsoft Visual Studio .NET 2003\Vc7\bin\vcvars32.bat"
nvcc -I"C:\Python25\include" -ccbin "C:\Program Files\Microsoft Visual Studio .NET 2003\Vc7\bin" "-IC:\Program Files\Microsoft Visual Studio 8\VC\include" -arch compute_11 -c testchagpp.cu
rem nvcc -c testchagpp.cu -I"C:\Python25\include" -I"C:\Program Files\Microsoft Visual Studio .NET 2003\Vc7\PlatformSDK\Include" -ccbin "C:\Program Files\Microsoft Visual Studio .NET 2003\Vc7\bin" -arch compute_11