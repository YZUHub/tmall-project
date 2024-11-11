# tmall-project

This repository tracks the workloads for Tmall customer data analysis project for the Data Analysis course for senior students 2024.

## Requirements

You will need to following tools to participate in this project:

* GitHub Account
* Python 3.9

## How to initialize your workspace

Run the following commands sequentially to get started with your project:

```
git clone https://github.com/YZUHub/tmall-project.git
cd tmall-project
git checkout submit-<your student ID>
git checkout -b "DA-<Student_ID>-<give a name to your working branch>"
```

Then create and activate your virtual environment depending on the OS and terminal you are using. Then use the following commands:

```
pip install poetry
poetry install
```

This will setup the working directory with necessary packages installed in your python virtual environment.

## How to do project works

This repository comes with a CLI for you to train or verify you models. You will add your codes, inside the `workflow` directory having a `__init__.py` containing `train_model` and `validate_model` functions.

* Train your model by running `python manage.py train`, this will invoke the `train_model` function you put inside `workflow/__init__.py`. Please save the model(s) you train inside the `models` directory as *.pkl* files.
* Inside the `verify_model` function, add logics to load the model you trained that was stored inside the `models` directory. After loading the model, there should be logics to test your model using a sample of test data.

You can create any number of modules you want inside the workflow directory. **Remember that the root of your project will be the root directory of this repository.** Because you will run the CLI commands from the root directory of this project.

You can do your experiments using `Jupyter Notebooks`. In that case, please save your notebooks inside the `notebooks` directory.

## How to install new dependencies

Please use the following command to install a new package:

```
poetry add <package_name>
```

## How to submit project

After you have completed your project, create a new `Pull Request` from the branch you were working on (for example, `DA-MH24108-testrun5`) to your submit branch (for example, `submit-MH24108`).
