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


## cloud setup

See Configuring a D-Wave System section in [LINK](https://dwave-meta-doc.readthedocs.io/en/latest/overview/dwavesys.html)

In the virtual environment do:

1. install dwave-ocean-sdk: `pip install dwave-ocean-sdk`
2. Then in terminal run: `dwave config create`
3. Follow props and copy the authentication token from Leap (found on LEAP dashboard).
4. Can test by running: `dwave ping` in the terminal OR more expensive: `dwave sample --random-problem`

Annoyingly the leap access seems to now be through their [quantum-launchpad](https://www.dwavesys.com/quantum-launchpad/)
