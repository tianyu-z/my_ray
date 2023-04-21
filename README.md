# How to start working on this?
```
git clone https://github.com/tianyu-z/my_ray.git
# switch to the branch
cd my_ray
git checkout IPD_2.2.0

# env
conda env create -f environment.yml # I recommand to use this because I encountered some other bug when I try to install everything from scrach

# copy the modified ray files to the package (cannot directly install the ray from source, I tried)
# Please change the following env path to yours
cp my_ray/rllib/examples/env/utils/mixins.py home/work/miniconda3/envs/ray38/lib/python3.8/site-packages/ray/rllib/examples/env/utils/mixins.py
cp my_ray/rllib/examples/env/matrix_sequential_social_dilemma_messaged.py home/work/miniconda3/envs/ray38/lib/python3.8/site-packages/ray/rllib/examples/env/matrix_sequential_social_dilemma_messaged.py
# run
python my_ray/rllib/examples/env/utils/iterated_prisoners_dilemma_env_customize.py 
```
# Changes made
New files:
- `iterated_prisoners_dilemma_env_customize.py` at `rllib/examples/iterated_prisoners_dilemma_env_customize.py`
- `matrix_sequential_social_dilemma_messaged.py` at `rllib/examples/env/matrix_sequential_social_dilemma_messaged.py`
- `visualize.ipynb` at root folder to visualize the output `csv` logging file.

Updated files:
`mixins.py` at `rllib/examples/env/utils/mixins.py`

