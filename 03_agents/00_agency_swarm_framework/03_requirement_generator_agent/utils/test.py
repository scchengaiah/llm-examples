text = """This is a string with leading and trailing whitespace.
            Another line of data."""
stripped_text = '\n'.join(line.strip() for line in text.splitlines())

print(stripped_text)

import time

# Start time
start_time = time.time()

time.sleep(5)

# End time
end_time = time.time()

# Calculate and print the time taken
time_taken = end_time - start_time
print(f"Time taken: {int(time_taken)} seconds")