import json
import numpy as np
import matplotlib.pyplot as plt

# Read JSON files
with open('logs/robomimic/XArm7Env-v1/passthrough_execution_time.json', 'r') as file:
    passthrough_data = json.load(file)

with open('logs/robomimic/XArm7Env-v1/teleop_execution_time.json', 'r') as file:
    teleop_data = json.load(file)

# Extract execution time values
passthrough_execution_times = list(passthrough_data.values())
teleop_execution_times = list(teleop_data.values())

# Convert execution times to minutes (if they are not already in minutes)
passthrough_execution_times = [time * 60 for time in passthrough_execution_times]
teleop_execution_times = [time * 60 for time in teleop_execution_times]

# Calculate mean and standard deviation
passthrough_mean = np.mean(passthrough_execution_times)
teleop_mean = np.mean(teleop_execution_times)

passthrough_std_dev = np.std(passthrough_execution_times)
teleop_std_dev = np.std(teleop_execution_times)

# Set custom colors
box_colors = ['lightblue', 'lightgreen']

# Plot box plot with custom colors and grid
plt.boxplot([passthrough_execution_times, teleop_execution_times], labels=['Passthrough', 'Teleop'], patch_artist=True,
            boxprops=dict(facecolor=box_colors[0]), medianprops=dict(color='black'))
for box, color in zip(plt.boxplot([passthrough_execution_times, teleop_execution_times], labels=['Passthrough', 'Teleop'], patch_artist=True)['boxes'], box_colors):
    box.set_facecolor(color)

plt.ylabel('Execution Time (seconds)')
plt.title('Box Plot of Execution Time')
plt.grid(True, linestyle='--', alpha=0.7)

# Print mean and standard deviation
print('Passthrough Mean Execution Time:', passthrough_mean, 'seconds')
print('Passthrough Standard Deviation:', passthrough_std_dev, 'seconds')

print('\nTeleop Mean Execution Time:', teleop_mean, 'seconds')
print('Teleop Standard Deviation:', teleop_std_dev, 'seconds')

plt.show()
