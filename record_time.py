"""
Writes day of the week and time to a file.

Script written for crontab tutorial.

Author: Jessica Yung 2016

"""
import time

filename = "record_time.txt"

# Records time in format Sun 10:00:00
current_time = time.strftime('%a %H:%M:%S')

# Append output to file. 'a' is append mode.
with open(filename, 'a') as handle:
	# Write (Append) output to a line
    handle.write(str(current_time))
    # Newline to separate different lines of output
    handle.write('\n')
