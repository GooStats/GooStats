- I followed your instructions and I still cannot even run the naive-Reactor example. 
	- Try to use cuda 7.5 + gcc 4.9 and run on an nVidia GPU. 
	- Your GPU cannot be too old, at least support sm_20, otherwise function pointers are not supported. Then you need to rewrite GoOFit. Look at [List of Nvidia GPUs](https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units) and search your GPUs. For example, Tesla C1060/M1060, produced on 2009, is sm1.3, you cannot run GooFit on it, and thus you cannot use GooStats. 
	- If you have a new GPU and you still cannot run, check your GPU driver version. It should be new enough to support compute_20.
	- For more information about installation, please consult Google/StackOverflow and [nVidia documents](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) and your GPU cluster administrator.
	
- I got zero NLL
	- I guess you have written a customized PDF. Check your customized PDF. Did you put/load indices in correct order? Did you get the parameters correctedly? Did you put constants correctedly?
