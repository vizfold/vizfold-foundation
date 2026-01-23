# Setup issues

The purpose of this guide is to help possible users fix certain problems that may arise when attempting to set up or run Vizfold.
In general make sure when setting up or running the program make sure you have started an interactive job first and that you are on that job's clusters.

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

# Runtime Issues

## Jackhmmer issues

### Description:

You may encounter an error involving Jackhmmer with the description: failed with a error while loading shared libraries: libopenblas.so.0: cannot open shared object file: No such file or directory

### Workarounds/Solutions: 

Simply install libopenblas:
```
mamba install -c conda-forge libopenblas
```

# Issues from running the web app under PR #12

## Port Forwarding Issue

### Description:

If you are running the web app from Visual Studio Code, it should automatically forward the port when you run "python app.py". However, this is not always the case, as the forwarded localhost can fail to load.

### Workarounds/Solutions:

If you are unable to start the web app by using "python app.py", run the following instead:
```
export FLASK_APP=app.py
# This will also kill any previous flask process that was running
flask run --host=0.0.0.0 --port=9000 > flask.log 2>&1 &
echo $! > flask.pid
```
This starts the web server while also letting you use the current terminal for further debugging if needed. 

Next run the following command:
```
hostname
```

Then, on another terminal, run the following command to forward the port using ssh:
```
ssh -L 9000:[result of hostname]:9000 [your username]@login-ice.pace.gatech.edu
```

Then after that you can go to http://localhost:9000

### General Note:

You can check if the process is actually running by running in the terminal:
```
curl http://localhost:9000
```