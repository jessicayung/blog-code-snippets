"""
Writes day of the week and time to a file.

Script written for crontab tutorial.

Author: Jessica Yung 2016

"""
import time

filename = record_time.txt

# Records time in format Sun 10:00:00
current_time = time.strftime('%a %H:%M:%S')

# Append output to file. 'a' is append mode.
with open(filename, 'a') as handle:
	# Write (Append) output to a line
    handle.write(str(html))
    # Newline to separate different lines of output
    handle.write('\n')
    # Extra newline to visually differentiate different lines of output
    # if you're using an editor with text wrapping or just want the
    # extra space
    handle.write('\n')
