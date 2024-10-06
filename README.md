# Guided Stream of Search

This is the code for the paper

## Environment settings

```bash
conda env create --name countdown --file environment.yaml
conda activate countdown
cd stream-of-search
pip install -r requirements.txt
cd ..
cd tril
pip install -e .
pip install flash-attn --no-build-isolation
```

> [!NOTE]  
> Please do not modify the package versions. Any changes may cause numerical instability as discussed in [this article](https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo).

## Computational resources

- Training: 4 x NVIDIA A100 80GB 
- Inference: 1 x NVIDIA RTX 3090

## Prerequisite

The base directory is set to `/home/{user}/guided-stream-of-search`. All data, checkpoints, and other files will be stored under this base directory. Please update this path as needed before running the script.


## Data generation

The data can be found in `/home/{user}/guided-stream-of-search/stream-of-search/data`.

```bash
conda activate countdown
cd stream-of-search
sh script/task/gen_task.sh  # Training
sh script/task/gen_task_final.sh  # Evaluation
```

## Unsupervised pre-training

The checkpoint can be found in `/home/{user}/guided-stream-of-search/stream-of-search/output`.

```bash
conda activate countdown
cd stream-of-search
sh script/gpt2/train_sos.sh
```

## Supervised fine-tuning with self-generated data

The data and checkpoints can be found in `/home/{user}/guided-stream-of-search/stream-of-search/output`

```bash
conda activate countdown
cd stream-of-search

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

## RL fine-tuning

The checkpoints can be found in `/home/{user}/guided-stream-of-search/tril/output`

```bash
conda activate countdown
cd tril
sh examples/countdown/countdown_ppo_op.sh
```

## Evaluation

The results can be found in the checkpoint directory you provided.

### Unsupervised Pre-training & Supervised Fine-tuning

```bash
conda activate countdown
cd stream-of-search
python eval.py --ckpt {ckpt} --start 0
...
python eval.py --ckpt {ckpt} --start 10000
cd ..
python summary.py --ckpt {ckpt} 
```

### RL fine-tuning

```bash
conda activate countdown
cd tril
python eval.py --ckpt {ckpt} --start 0
...
python eval.py --ckpt {ckpt} --start 10000
cd ..
python summary.py --ckpt {ckpt} 
```

## Acknowledgements

- https://github.com/kanishkg/stream-of-search/
- https://github.com/Cornell-RL/tril
