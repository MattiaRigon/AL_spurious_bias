This project is build on top of the work ["Assessing and enhancing robustness of active learning strategies to spurious bias"](https://openreview.net/forum?id=2XVECaYiFB), in particulary starting from the repository: ["repository"](https://anonymous.4open.science/r/jSYaklvh/README.md).


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

### RRR

For the waterbirds dataset are already provided the explanations masks, while with few modifications at `explanation.py` is possible to compute the masks for another dataset. In order to do it you must install LangSAM following: ["instructions"](https://github.com/paulguerrero/lang-sam?tab=readme-ov-file#installation) and have already stored the dataset locally, then you should change the prompt that you want to use in order to generate the masks.

In order to active the rrr loss inside `conf/active.yaml` there is a rrr boolean variable that can be activated or disactivated putting it to True or False.

## Acknowledgements
Most of the code is based on the [Baal](https://github.com/baal-org/baal) and [SubpopBench](https://github.com/YyzHarry/SubpopBench).
The implementation of `BADGE` and `BAIT` is based on the repo [badge](https://github.com/JordanAsh/badge) and `Cluster-Margin` is based on the repo [deep-active-learning](https://github.com/cure-lab/deep-active-learning)


