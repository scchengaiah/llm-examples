import asyncio
from contextvars import ContextVar

# This is a context variable that will have different values in different async contexts.
request_id = ContextVar('request_id', default=None)

class MyClass:
    def __init__(self, name):
        self.name = name

    async def process(self):
        # Access the context variable for this task.
        if self.name in ('B', 'D'):
            await asyncio.sleep(3) # Sleep for some time for specific tasks to understand the context retrieval.
        current_request_id = request_id.get()
        print(f"Processing {self.name} in request {current_request_id}")

async def main():
    # Simulate two different request contexts.
    request_id.set('req-A')
    instance_a = MyClass('A')
    task_1 = asyncio.create_task(instance_a.process())

    request_id.set('req-B')
    instance_b = MyClass('B')
    task_2 = asyncio.create_task(instance_b.process())

    request_id.set('req-C')
    instance_c = MyClass('C')
    task_3 = asyncio.create_task(instance_c.process())

    request_id.set('req-D')
    instance_d= MyClass('D')
    task_4 = asyncio.create_task(instance_d.process())

    # Run both instances' process method concurrently.
    await asyncio.gather(task_1, task_2, task_3, task_4)

asyncio.run(main())