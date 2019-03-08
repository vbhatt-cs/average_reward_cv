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
 $ conda env create -f requirements.yml
 $ conda activate average_reward_cv
 ```

## Running
* After conda environment is activated, run `python main.py`.
* Run `python main.py --help` for documentation about optional parameters.

## Files
* _main.py_ - Main script to run the experiments
* _gridworld.py_ - Class for grid world environment
* _algs.py_ - Implementation of algorithms

## TODO
- [x] Implement and test environments
- [ ] Implement algorithms for episodic case and verify
    - [ ] Scaling TD
    - [ ] Scaling target
    - [ ] Per decision
    - [ ] ACV
- [ ] Implement algorithms for continuing case
    - [ ] Scaling TD
    - [ ] Scaling target
    - [ ] Per decision
    - [ ] ACV for Q only
    - [ ] ACV for Q, R
- [ ] Integrate MLFlow for running experiments
- [ ] Plot graphs