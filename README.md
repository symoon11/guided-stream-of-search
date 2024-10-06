# Guided Stream of Search

This is the code for the paper

## Environment settings

```
conda env create --name countdown --file environment.yaml
conda activate countdown
cd stream-of-search
pip install -r requirements.txt
cd ..
cd tril
pip install -e .
pip install flash-attn --no-build-isolation
```

## Computational resources
- Training: 4 x NVIDIA A100 80GB 
- Inference: 1 x NVIDIA RTX 3090

## Prerequisite

The base directory is set to `/home/{user}/guided-stream-of-search`. All data, checkpoints, and other files will be stored under this base directory. Please update this path as needed before running the script.


## Data generation

The data can be found in `/home/{user}/guided-stream-of-search/stream-of-search/data`.

```
conda activate countdown
cd stream-of-search
sh script/task/gen_task.sh  # Training
sh script/task/gen_task_final.sh  # Evaluation
```

## Unsupervised pre-training

The checkpoint can be found in `/home/{user}/guided-stream-of-search/stream-of-search/output`.

```
sh script/gpt2/train_sos.sh
```

## Supervised fine-tuning with self-generated data

The data and checkpoints can be found in `/home/{user}/guided-stream-of-search/stream-of-search/output`

```
# Iteration 1
sh script/gpt2/iter1/gen_star_s0.sh
sh script/gpt2/iter1/gen_gsos_rand_s0.sh --start 0
...
sh script/gpt2/iter1/gen_gsos_rand_s0.sh --start 199000
sh script/gpt2/iter1/train_gsos_rand_s0.sh

# Iteration 2
sh script/gpt2/iter2/gen_gsos_rand_s0.sh --start 0
...
sh script/gpt2/iter2/gen_gsos_rand_s0.sh --start 199000
sh script/gpt2/iter2/train_gsos_rand_s0.sh

# Iteration 3
sh script/gpt2/iter3/gen_gsos_rand_s0.sh --start 0
...
sh script/gpt2/iter3/gen_gsos_rand_s0.sh --start 199000
sh script/gpt2/iter3/train_gsos_rand_s0.sh
```

> [!NOTE]  
> The data generation process requires a large number of GPUs. It is recommended to use over 40 RTX 3090 GPUs and run the scripts in parallel.