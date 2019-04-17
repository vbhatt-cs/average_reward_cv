# Analysis of Value-Based Algorithms in Average Reward Setting
Code for the project on various experiments with algorithms for average reward setting. See _CMPUT609_Project.pdf_ for the report.

## Requirements
- numpy
- gym
- matplotlib
- pandas

## Installation
* Install [anaconda with python 3](https://www.anaconda.com/distribution/).
* Create a conda environment and activate it:
 ```
 $ conda env create -f requirements.yaml
 $ conda activate average_reward_cv
 ```

## Running
* After conda environment is activated, run `python main.py` for a single run.
* Run `python main.py --help` for documentation about optional parameters.
* _run_experiments.py_ contains code to run experiments and log the results. Uncomment the required experiments in the file and run `python run_experiments.py`

## Files
* _main.py_ - Main script to run the algorithms
* _envs.py_ - Implementation of environments
* _algs.py_ - Implementation of algorithms
* _features.py_ - Feature extraction classes
* _policies.py_ - Implementation of policies
* _tiles3.py_ - Tile Coding Software by Rich Sutton (http://incompleteideas.net/tiles/tiles3.html)
* _true_value.py_ - Code to calculate true value function for prediction problems
* _run_experiments.py_ - File to run the experiments and log results
* _plotting.py_ - File containing plotting functions. Code is not documented and is used specifically for the figures in report