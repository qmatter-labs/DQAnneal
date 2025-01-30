# DQAnneal




Note for HPC usage can require certain gcc version. The following conda code can help

```
conda create -n dwave310 python=3.10

## update g++
conda install -c conda-forge gxx_linux-64=11

## check
conda list

### check for gcc_linux-64 11.4.0
## make sure conda can find it! 

## IF and only IF there are problems can use: 
export CC=$(conda info --base)/envs/myenv/bin/x86_64-conda-linux-gnu-gcc 
export CXX=$(conda info --base)/envs/myenv/bin/x86_64-conda-linux-gnu-g++
## (but should NOT be necessary!)
```
