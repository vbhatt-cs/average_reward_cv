# Control variates for average reward
Using control variates for off-policy learning in average reward setting.

Extension of De Asis, Kristopher, and Richard S. Sutton. "Per-decision Multi-step Temporal Difference Learning with Control Variates." arXiv preprint arXiv:1807.01830 (2018).

## Requirements
- numpy
- gym
- matplotlib

## Installation
* Install [anaconda with python 3](https://www.anaconda.com/distribution/).
* Create a conda environment and activate it
 ```
 $ conda env create -f requirements.yaml
 $ conda activate average_reward_cv
 ```

## Running
* After conda environment is activated, run `python main.py`.
* Run `python main.py --help` for documentation about optional parameters.
* To run all experiments, run `python run_experiments.py`

## Files
* _run_experiments.py_ - File to run the experiments and log results
* _main.py_ - Main script to run the algorithms
* _envs.py_ - Implementation of environments
* _algs.py_ - Implementation of algorithms
* _features.py_ - Feature extraction classes
* _policies.py_ - Implementation of policies
* _tiles3.py_ - Tile Coding Software by Rich Sutton (http://incompleteideas.net/tiles/tiles3.html)
* _true_value.py_ - Code to calculate true value function in gridworld environment

## TODO
- [x] Implement and test environments
- [x] Implement algorithms for continuing case
    - [x] On policy
        - [x] N-step prediction
        - [x] Lambda prediction
        - [x] N-step control
        - [x] Lambda control
    - [x] One-step/full Rbar update
    - [x] R-Learning
    - [x] Off policy
        - [x] N-step prediction
        - [x] Lambda prediction
        - [x] N-step control
        - [x] Lambda control
        - [x] Add a testing phase to test off-policy control
    - [x] CV
- [x] Set up experiment pipeline
- [ ] Set up plotting pipeline
- [x] Plot graphs
    - [x] Prediction (on-policy)
        - [x] RMS Error with n-step
        - [x] RMS Error with lambda
        - [x] RMS Error with full rbar or one step rbar
    - [x] Prediction (off-policy)
        - [x] RMS Error with n-step
        - [x] RMS Error with lambda
        - [x] RMS Error with full rbar or one step rbar
    - [x] Prediction (cv)
        - [x] RMS Error with n-step
        - [x] RMS Error with lambda
        - [x] RMS Error with full rbar or one step rbar
        - [x] RMS Error without cv for rbar    
    - [x] Control
        - [x] Reward vs episodes for cv and no cv
        - [x] Fully off-policy (like random behaviour, test with greedy?)
- [x] Random walk env
- [ ] Random walk experiments
    - [ ] One step vs multi-step (with different init_rbar)
    - [ ] Full rbar vs one step rbar