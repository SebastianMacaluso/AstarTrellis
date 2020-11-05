# hierarchical-trellis
data structures, algorithms, and experiments for MAP and Z computation for distribution of hierarchical clustering

## New code setup

Run either

```
python setup.py install
```

or

```
export PYTHONPATH=`pwd`:$PYTHONPATH
```

## Running A Star on a SLURM cluster

There is a yaml file [bin/a_star/cc_aloi.yaml] that we will use.

First run:

```
wandb sweep bin/a_star/cc_aloi.yaml
```

This gives output like:

```
wandb: Creating sweep from: bin/a_star/cc_aloi.yaml
wandb: Created sweep with ID: n25yz7k8
wandb: View sweep at: https://app.wandb.ai/nmonath/AStar/sweeps/n25yz7k8
wandb: Run sweep agent with: wandb agent nmonath/Astar/n25yz7k8
```

Then run

```
# sh bin/launch_sweep.sh sweepId numMachines numCpuPerMachine mbRAMperMachine
sh bin/launch_sweep.sh nmonath/Astar/n25yz7k8 10 1 5000
```