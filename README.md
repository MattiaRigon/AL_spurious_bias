Repository for the TMLR submission ["Assessing and enhancing robustness of active learning strategies to spurious bias"](https://openreview.net/forum?id=2XVECaYiFB).


## Installation
To install the necessary dependencies, run:
```sh
pip install -r requirements.txt
```

## Running the experiments
To run the experiments, use the following command:
```sh
python main.py -cn active dataset=DATASET heuristic=AL_METHOD
```
The list of available datasets and active learning methods can be found in the `conf/dataset` and `conf/heuristic` directories, respectively.
For other configuration options, please refer to the `conf/active.yaml` file.
Our method `DIAL` is named `QBC` in the configuration files.

## Acknowledgements
Most of the code is based on the [Baal](https://github.com/baal-org/baal) and [SubpopBench](https://github.com/YyzHarry/SubpopBench).
The implementation of `BADGE` and `BAIT` is based on the repo [badge](https://github.com/JordanAsh/badge) and `Cluster-Margin` is based on the repo [deep-active-learning](https://github.com/cure-lab/deep-active-learning)


