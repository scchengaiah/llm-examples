# Aider

Aider lets you pair program with LLMs, to edit code in your local git repository. It leverages the power of large language models to assist you in writing, refactoring, and understanding code. With Aider, you can streamline your development process, reduce errors, and improve code quality by getting real-time suggestions and corrections from an AI assistant.

## Installation

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

## Configuration

In this example, we are going to use `aider` with `Azure OpenAI` subscription. For complete list of LLM support, refer to this [link](https://aider.chat/docs/llms.html).

Set the following environment variables.

```cmd
set AZURE_API_KEY=<your-api-key>
set AZURE_API_VERSION=<your-api-version>
set AZURE_API_BASE=<your-api-base>
```

Launch aider with the following command disabling auto commits.

```bash
aider --model azure/<your_deployment_name> --no-auto-commits false --no-dirty-commits false
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

## Usage

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
