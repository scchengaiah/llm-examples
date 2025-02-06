from upsonic import Agent, Task
from upsonic.client.tools import Search

task = Task("Who developed you?")

agent = Agent("Coder")

agent.print_do(task)
