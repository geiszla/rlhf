# Machine Learning Project

This is the repository for the AI Center Projects in Machine Learning Research course in 2023 Spring.

<!-- TODO: Remove conda-lock if not needed (change readme to simplify Anaconda setup) -->

## Setting up

*Note: if you are using Windows, you will need to either use a Linux VM or WSL (see [WSL setup instructions](#setting-up-wsl-recommended-for-windows) below). In both cases you will need to use `linux` as your OS identifier (e.g., for the Anaconda lock file name).*

### Setting up WSL (recommended for Windows)

1. Set up [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
1. Open WSL command line and clone the repo using it (to a path NOT starting with `/mnt/`) instead of using a Windows command prompt or PowerShell. This will make the development environment faster.
1. Add `export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0` to your shell profile (e.g., to the end of `~/.bashrc` or `~/.zshrc` or similar) to enable WSL to open windows and display GUIs.
1. If you are using VSCode, run `code .` inside the project directory to open it (or if you've opened the project before, you can access it from `File -> Open Recent`). See [Open a WSL project in Visual Studio Code](https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-vscode#open-a-wsl-project-in-visual-studio-code) for more details.
1. Do all further setup inside the WSL command line or from the terminal of VSCode opened from WSL.

### Installing dependencies

1. Install [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (Mamba, Anaconda, or Miniconda also works, but is not necessary)
1. Add `alias mamba='micromamba'` to your shell profile (e.g., to the end of `~/.bashrc` or `~/.zshrc` or similar) to be able to use the `mamba` command.
1. Navigate to the project directory, then run `mamba create --name ml_project --file conda-<linux/osx>-64.lock` (replace `<linux/osx>` with your OS) to create the Mamba environment.
1. Run `mamba activate ml_project` to activate the virtual environment.
1. Run `poetry install` in the project directory to install dependencies of the project.
1. To install MuJoCo, follow the [instructions in the GitHub repo](https://github.com/openai/mujoco-py/#install-mujoco).
1. Add `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin` to your shell profile and start a new shell to make MuJoCo discoverable.
1. Make sure all the required libraries are installed by running `sudo apt install gcc libosmesa6-dev libgl1 libglfw3 patchelf`
1. If you don't have `ffmpeg` installed yet, [install it on your system](https://ffmpeg.org/download.html) or run `pip install imageio-ffmpeg` to install it in the project locally.
1. Run `pre-commit install` to install pre-commit hooks (they will run some checks before each commit to the repo).

### Setting up VSCode (recommended)

1. Install and open [VSCode](https://code.visualstudio.com/download)
1. Install these VSCode extensions (by searching for them on the extensions tab): `charliermarsh.ruff`, `njpwerner.autodocstring`, `visualstudioexptteam.vscodeintellicode`, `ms-python.black-formatter`, `ms-python.isort`, `ms-python.vscode-pylance`, `ms-python.pylint`, `ms-python.python`, `kevinrose.vsc-python-indent`, `tamasfe.even-better-toml`
1. Open the command palette, choose `Python: Select Interpreter`, then select the virtual environment created by Micromamba.
   *Note: If the desired environment is not in the list, you can find the location of the environments by running `mamba env list`, then add the interpreter as a new entry.*
1. Start a new terminal. VSCode will automatically activate the selected environment.

*Note: for now VSCode does not use the selected interpreter for Git commands (see [issue](https://github.com/microsoft/vscode-python/issues/10165)), so you need to create commits from changes that contain Python code from the command line (pre-commit hooks need to run from the right Python environment).*

### Without VSCode

1. Run `mamba activate ml_project` to activate the Mamba environment.
1. It's recommended to set up the extensions in your IDE equivalent to those listed above in the VSCode setup section for a more convenient development.

## Downloading the expert model

To download the expert model, run `python -m rl_zoo3.load_from_hub --algo sac --env HalfCheetah-v3 -orga sb3 -f logs/` after activating the mamba environment.

## Managing dependencies

- To **add** a package to dependencies, use `poetry add <package>` for dependencies or `poetry add -D <package>` for development dependencies
- If adding a package using Poetry fails (or if a version from Anaconda is needed), add the package and its version to `environment.yml`, then run:
  - `conda-lock -k explicit --conda mamba` to update the Anaconda lock file
  - `mamba update --file conda-<linux/osx>-64.lock` (replace `<linux/osx>` with your OS) to update the packages in your current environment
- To **remove** a package from the dependencies, use `poetry remove <package>` or remove them from `environment.yml`. In the latter case, you will also need to run the commands above to update the Anaconda lock files and current environment packages.

## Running the code

You can run scripts specified in `pyproject.toml` with `poetry run <script name>`. For example, to run the `train` script, run `poetry run train` (you might also need to run `poetry install` before to update the dependencies).

To run individual scripts, use `python -m <module path>`. E.g., if you want to run the script `ml_project/reward_model/pretrain_reward_model.py`, run `python -m ml_project.reward_model.pretrain_reward_model`.

### Reproducing the results

1. Run `poetry run train` to train the baseline RL agents (you can change the algorithm by editing `ml_project/rl/train_rl_agent.py`)
1. Run `poetry run generate` to generate videos for the trained agent (you can change the algorithm by editing `ml_project/rl/generate_videos_and_data.py`)
1. Run `poetry run flask --app ml_project/web_interface run` to start the web interface to collect preferences. Preferences are written in `output/preferences.csv`.
1. Run `poetry run generate_reward` to generate rewards for observations on trajectories made by the trained RL agent (you can change the algorithm by editing `ml_project/rl/generate_reward_data.py`).
1. Run `poetry run pretrain_reward` to pretrain the reward model on the generated reward data.
1. Run `poetry run create_dataset` to create a data set using the collected preferences.
1. Run `poetry run finetune_reward` to finetune the reward model using the preferences collected.
1. Run `poetry run tensorboard --logdir lightning_logs` to start the tensorboard to see train and validation loss.
