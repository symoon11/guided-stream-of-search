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

## Prerequisite

The base directory is set to `/home/{user}/guided-stream-of-search`. All data, checkpoints, and other files will be stored under this base directory. Please update this path as needed before running the script.


## Data generation

The data can be found in `/home/{user}/guided-stream-of-search/stream-of-search/data`.

```
conda activate countdown
cd stream-of-search
sh script/task/gen_task.sh  # For training
sh script/task/gen_task_final.sh  # For evaluation
```
