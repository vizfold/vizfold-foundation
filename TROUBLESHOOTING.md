# Setup issues

The purpose of this guide is to help possible users fix certain problems that may arise when attempting to set up or run Vizfold.

## Running out of storage while creating the environment

### Description:

Setting up the environment requires the download of many large dependencies which can quickly fill up the quota of the home directory on the HPC cluster.

### Workarounds/Solutions:

Find your designated space under /storage in the HPC cluster and set up your environment in there. On the GT HPC cluster the directory will look like:
```
/storage/ice1/X/Y/$USER
```
where X and Y are numbers that may be different for each user. You can find your directory by running the command
```
find /storage -maxdepth 5 -type d -user $USER -ls 2>/dev/null
```

Make sure CONDA_PKGS_DIRS and ENV_PREFIX are set correctly. Run the following:
```
export SCRATCH_ROOT=/storage/ice1/<X>/<Y>/$USER
# Replace X and Y with the numbers of your directory
mkdir -p "$SCRATCH_ROOT/conda-pkgs" "$SCRATCH_ROOT/envs"
export CONDA_PKGS_DIRS="$SCRATCH_ROOT/conda-pkgs"
export ENV_PREFIX="$SCRATCH_ROOT/envs/openfold_env"
```

To clear any cache/dependency files left by conda/mamba in your home directory run
```
mamba clean -a
```
## "No module named torch"

### Description:

This error has happened when trying to create a virtual environment using the environment.yml file, and while trying to run the setup.py file using "pip install -e ." The error occurs while trying to import the module "flask-attn".

### Workarounds/Solutions:

If you were creating a virtual environment with the environment.yml file:
- The environment has likely still been created anyway. Activate it and manually pip install the dependencies listed under the "pip" section of the environment.yml file.
- To install flash-attn run the following command: 
```
python -m pip install --no-cache-dir flash-attn==2.6.3 --no-build-isolation
```

If you were running setup.py
- Run the following instead of "pip install -e .":
```
python setup.py develop
```

Both these solutions will install flash-attn without using an isolated build environment where pytorch may not be accessible

# Issues from running the web app under PR #12
