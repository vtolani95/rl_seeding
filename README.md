RL Seeding for Viral Marketing
==========
### Setup
1. Use with python3, install dependencies into a virtual environment.
```
python3 -m venv ../$PWD/venv-mpc
source venv-mpc/bin/activate
pip3 install -U pip
pip3 install -r requirements.txt
pip3 install tensorflow 
```
2. Install the following dependencies into a virtual environment.
```
pip3 install -e envs-vmpc/
Install openAI baselines (see https://github.com/openai/baselines)
```

To run the script:
```
#random trainer
PYTHONPATH='.' python train.py --logdir_prefix 'output/' --config_name 'randomV0_64_200_10_0x5_10_0_0_0.6e5.0_0_200' --logdir_suffix '_test'

#greedy trainer
PYTHONPATH='.' python train.py --logdir_prefix 'output/' --config_name 'greedyV0_64_200_10_0x5_10_0_0_0.6e5.0_0_200' --logdir_suffix '_test'

#DQN Method
PYTHONPATH='.' CUDA_VISIBLE_DEVICES='0' python train.py --logdir_prefix 'output/' --config_name 'dqnV0_64_200_10_0x5_10_0_0_0.5en4_1e7_5e4_1en1_0x02_10_32_1000_1x0_1e4_0.0_0_200' --logdir_suffix '_test'
```
