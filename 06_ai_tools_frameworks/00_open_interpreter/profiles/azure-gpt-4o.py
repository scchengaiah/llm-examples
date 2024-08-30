from interpreter import interpreter

interpreter.llm.model = "azure/gpt-4o"
interpreter.llm.context_window = 110000
interpreter.llm.max_tokens = 4096
interpreter.llm.supports_functions = True
interpreter.llm.supports_vision = True

interpreter.anonymized_telemetry = False
interpreter.loop = True
# Set this to False if the actions performed will be destructive or run the Interpreter within dockerized environment.
interpreter.auto_run = True
interpreter.verbose = True
interpreter.computer.import_computer_api = True


