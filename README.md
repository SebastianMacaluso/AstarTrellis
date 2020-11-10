# A star & Trellis Algorithm for Hierarchical Clustering
Data structures, algorithms, and experiments for MAP and Z computation for hierarchical clustering

## New code setup

Install package

```
make install
```

## Running a grid search with wandb sweep process
 If the dir for wandb is inside the package, paths could be relative, else set absolute paths.
1. YAML file settings, e.g. [`Ginkgo.yaml`](bin/a_star).  
	- Set path to executable under `program`, e.g. for Ginkgo jets `[path to file/run_a_star_iter_ginkgo.py]`
	- Set the dir for wandb (`wandb_dir`)
	- Set the dir where the dataset is located (`dataset_dir`)
	- Set the dataset filename (`dataset`)
2. From the assigned directory for `wandb_dir`, run:
	- Install wandb if necessary: `pip install wandb`
	- `wandb login`
	- `wandb sweep [path to file/Ginkgo.yaml]`
	- There will be a terminal output as `wandb: Run sweep agent with: wandb agent sebas/Astar/l6de8bwm`
	  Run `wandb agent sebas/Astar/l6de8bwm`



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