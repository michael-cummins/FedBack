# Running Locally

All experiments are run through the main python file. For example, to run an experiment for FedBack using CIFAR-10 data with a communication rate of 10%, 

```
python main.py --back --cifar --rate 0.1
```

If you would like to use one of the other benchmarks, run one of the following,

```
python main.py --avg --cifar --rate 0.1
python main.py --prox --cifar --rate 0.1
python main.py --admm --cifar --rate 0.1
```

This will leave data for figures stored in the ```figure_data``` directory, which can then be plotted using,
```
python plotting.py
```
after all of the experiments have been ran.

Swapping out CIFAR-10 experiments for MNIST experiments can be done by simply replacing ```--cifar``` with ```--mnist```.

# Running In Cluster
These experiments require a massive amount of compute to complete so it is recommend you run them on a condor compute cluster if you want them finished in a reasonable time. To run all of the benchmarks for both CIFAR-10 and MNIST,


