- [Aider](#aider)
  - [Setup - Conda environment (Recommended)](#setup---conda-environment-recommended)
    - [Pre-requisites](#pre-requisites)
    - [Conda Configuration](#conda-configuration)
    - [Create Conda environment](#create-conda-environment)
    - [Update existing Conda environment](#update-existing-conda-environment)
    - [Delete existing Conda environment](#delete-existing-conda-environment)
    - [Aider Specific - Model metadata file](#aider-specific---model-metadata-file)
    - [Aider Specific - Model settings file](#aider-specific---model-settings-file)
    - [Launch Aider - Command Line mode](#launch-aider---command-line-mode)
    - [Launch Aider - Browser mode](#launch-aider---browser-mode)
  - [Setup - Standalone Virtual environment](#setup---standalone-virtual-environment)
    - [Installation](#installation)
    - [Configuration](#configuration)
    - [Usage](#usage)
  - [References](#references)
    - [Aider Capabilities](#aider-capabilities)


# Aider

Aider lets you pair program with LLMs, to edit code in your local git repository. It leverages the power of large language models to assist you in writing, refactoring, and understanding code. With Aider, you can streamline your development process, reduce errors, and improve code quality by getting real-time suggestions and corrections from an AI assistant.

## Setup - Conda environment (Recommended)

### Pre-requisites

Ensure that you have `conda` or `miniconda` installed in your environment and update the `PATH` to access `conda` executable from the command line.

To check if conda is already installed in your system. Execute the following command.

```bash
conda --version
```

### Conda Configuration

Create conda environment via yaml configuration file. A sample file can be found [here](./conda-env.yml)

### Create Conda environment

Create conda environment using the following command

```bash
conda env create --prefix D:/tmp/genai/venv/aider-conda-env -f conda-env.yml
```

If the prefix is not part of the `env_dirs`, then add the same.

```bash
conda config --add envs_dirs D:/tmp/genai/venv
```

### Update existing Conda environment

```bash
conda env update -f conda-env.yml
```

### Delete existing Conda environment

```bash
# Deleting a Conda Environment by Name
conda env remove --name my_env

# Deleting a Conda Environment by Path (Prefix)
conda env remove --prefix /path/to/your/env

# Verifying env deletion
conda env list
```

### Aider Specific - Model metadata file

Create a `.aider.model.metadata.json` file in one of these locations:

- Your home directory.
- The root if your git repo.
- The current directory where you launch aider.
- Or specify a specific file with the `--model-metadata-file <filename>` switch.

For this example, we will add the file `.aider.model.metadata.json` in the root of our git repo with the following content (Optional):

```json
{
  "azure/gpt-4o": {
    "max_tokens": 100000,
    "max_input_tokens": 128000,
    "max_output_tokens": 4096,
    "mode": "chat"
  }
}
```

### Aider Specific - Model settings file

Aider has a number of settings that control how it works with different models. These model settings are pre-configured for most popular models. But it can sometimes be helpful to override them or add settings for a model that aider doesn’t know about.

To do that, create a `.aider.model.settings.yml` file in one of these locations:

Your home directory.
The root if your git repo.
The current directory where you launch aider.
Or specify a specific file with the --model-settings-file <filename> switch.
If the files above exist, they will be loaded in that order. Files loaded last will take priority.

The yaml file should be a a list of dictionary objects for each model. For example, below are all the pre-configured model settings to give a sense for the settings which are supported.

Example content to place within the file:

```yaml
- accepts_images: false
  cache_control: false
  caches_by_default: false
  edit_format: whole
  editor_edit_format: null
  editor_model_name: null
  examples_as_sys_msg: false
  extra_params: null
  lazy: false
  name: gpt-3.5-turbo
  reminder: sys
  send_undo_reply: false
  streaming: true
  use_repo_map: false
  use_system_prompt: true
  use_temperature: true
  weak_model_name: gpt-4o-mini
- accepts_images: false
  cache_control: false
  caches_by_default: false
  edit_format: whole
  editor_edit_format: null
  editor_model_name: null
  examples_as_sys_msg: false
  extra_params: null
  lazy: false
  name: gpt-3.5-turbo-0125
  reminder: sys
  send_undo_reply: false
  streaming: true
  use_repo_map: false
  use_system_prompt: true
  use_temperature: true
  weak_model_name: gpt-4o-mini
```

### Launch Aider - Command Line mode

We use configuration file to launch `aider`. A sample config file can be found [here](./aider-config.yml). We copy this config file to our `%USERPROFILE%/aider-config.yml` or to the specific git repo where aider is used to refer within the command line. Edit this file based on your requirements.

For complete list of CLI options, refer [here](https://aider.chat/docs/config/options.html)

For complete list of YAML config options, refer [here](https://aider.chat/docs/config/aider_conf.html)

```bash
# We use Azure OpenAI with aider hence, before launching aider set the following environment variables. Check the documentation for other LLM providers.
# https://aider.chat/docs/llms.html
SET AZURE_API_KEY=<AZURE_API_KEY>
SET AZURE_API_BASE=<AZURE_API_BASE>
SET AZURE_API_VERSION=<AZURE_API_VERSION>

aider --chat-mode chat --config aider-config.yml 
```

### Launch Aider - Browser mode

```bash
# We use Azure OpenAI with aider hence, before launching aider set the following environment variables. Check the documentation for other LLM providers.
# https://aider.chat/docs/llms.html
SET AZURE_API_KEY=<AZURE_API_KEY>
SET AZURE_API_BASE=<AZURE_API_BASE>
SET AZURE_API_VERSION=<AZURE_API_VERSION>

aider --chat-mode chat --config aider-config.yml --browser
```


## Setup - Standalone Virtual environment

### Installation

We have included the required dependencies in [requirements.txt](./requirements.txt). For individual steps, refer to the below content:

- Create a virtual environment

```bash
python -m venv .venv
```

- Activate the virtual environment

```bash
.venv\Scripts\activate.bat
```

- Install Aider

```bash
pip install aider-chat
```

### Configuration

In this example, we are going to use `aider` with `Azure OpenAI` subscription. For complete list of LLM support, refer to this [link](https://aider.chat/docs/llms.html).

Set the following environment variables.

```cmd
set AZURE_API_KEY=<your-api-key>
set AZURE_API_VERSION=<your-api-version>
set AZURE_API_BASE=<your-api-base>
```

Launch aider with the following command disabling auto commits.

```bash
aider --model azure/<your_deployment_name> --no-auto-commits --no-dirty-commits
```

To update model specific configuration such as controlling token usage, follow the below steps. The complete configuration reference can be found [here](https://aider.chat/docs/config/options.html).

Create a `.aider.model.metadata.json` file in one of these locations:

- Your home directory.
- The root if your git repo.
- The current directory where you launch aider.
- Or specify a specific file with the `--model-metadata-file <filename>` switch.

For this example, we will add the file `.aider.model.metadata.json` in the root of our git repo with the following content:

```json
{
  "azure/gpt-4o": {
    "max_tokens": 100000,
    "max_input_tokens": 128000,
    "max_output_tokens": 4096,
    "mode": "chat"
  }
}
```

### Usage

Ask help related to aider using the following command. Make sure you launch aider using `aider --model azure/<your_deployment_name>` before executing any aider related command.

```bash
/help <ask-your-question-here>
```

Aider always work in context of files provided it. However, it also has the knowledge of the entire git repository related information. To add files to the aider context:

```bash
# Adding below files to the aider context provides it with the context to work upon.
/add main.py utils.py
```

For complete list of commands, refer [here](https://aider.chat/docs/usage/commands.html).

## References

[Aider-Docs](https://aider.chat/)

### Aider Capabilities

[Claude 3.5 and aider: Use AI Assistants to Build AI Apps](https://www.youtube.com/watch?v=0hIisJ3xAdU)

[ Aider and Claude 3.5: Develop a Full-stack App Without Writing ANY Code!](https://www.youtube.com/watch?v=BtAqHsySdSY)
