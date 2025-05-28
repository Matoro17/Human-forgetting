import os
from datetime import datetime

def log_message(log_filepath, message):
    """Logs a message to both console and a file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    try:
        with open(log_filepath, "a") as log_file:
            log_file.write(full_message + "\n")
    except Exception as e:
        print(f"[{timestamp}] Error writing to log file {log_filepath}: {e}")

