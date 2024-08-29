# Open Interpreter

A new way to use computers.

Open Interpreter lets language models run code.

You can chat with Open Interpreter through a ChatGPT-like interface in your terminal by running interpreter after installing.

This provides a natural-language interface to your computer’s general-purpose capabilities:

- Create and edit photos, videos, PDFs, etc.
- Control a Chrome browser to perform research
- Plot, clean, and analyze large datasets
- …etc.

## Setup

We have installed the following packages identified from this [Setup](https://docs.openinterpreter.com/getting-started/setup) page.

```bash
# Create virtual environment.
python -m venv .venv

# Install required packages.
pip install open-interpreter
pip install open-interpreter[os]
pip install open-interpreter[safe]
```

## Getting started

We are using Azure OpenAI subscription with `gpt-4o` as our LLM. We followed this [documentation](https://docs.openinterpreter.com/language-models/hosted-models/azure) to set appropriate configuration.

```bash
# Set the env variables.
set AZURE_API_KEY=
set AZURE_API_BASE=
set AZURE_API_VERSION=

# Launch Interpreter
interpreter --model azure/gpt-4o
```

## Advanced Usage

### Custom Profiles

We can start interpreter by pointing out to a specific profile that can contain customized configuration to suit a particular usecase.

It should be noted that the profiles can be configured as `yml` files or within the `py` file.

Refer [here](https://docs.openinterpreter.com/guides/profiles) to understand more about `Profiles`.

Refer [here](https://docs.openinterpreter.com/settings/all-settings) for complete settings.

## References

[Open Interpreter Docs](https://docs.openinterpreter.com/getting-started/introduction)
