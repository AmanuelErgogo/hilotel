import json
import numpy as np
import matplotlib.pyplot as plt

# Read JSON files
with open('passthrough_execution_time.json', 'r') as file:
    passthrough_data = json.load(file)

with open('teleop_execution_time.json', 'r') as file:
    teleop_data = json.load(file)

with open('pure_agent_execution_time.json', 'r') as file:
    pure_agent_data = json.load(file)

# Extract execution time values
passthrough_execution_times = list(passthrough_data.values())
teleop_execution_times = list(teleop_data.values())
pure_agent_execution_times = list(pure_agent_data.values())

# Calculate mean and standard deviation in seconds
passthrough_mean = np.mean(passthrough_execution_times) * 60
teleop_mean = np.mean(teleop_execution_times) * 60
pure_agent_mean = np.mean(pure_agent_execution_times) * 60

passthrough_std_dev = np.std(passthrough_execution_times) * 60
teleop_std_dev = np.std(teleop_execution_times) * 60
pure_agent_std_dev = np.std(pure_agent_execution_times) * 60

# Set custom colors
box_colors = ['lightblue', 'lightgreen', 'lightcoral']

# Plot box plot with custom colors and grid
plt.boxplot([passthrough_execution_times, teleop_execution_times, pure_agent_execution_times],
            labels=['Passthrough', 'Teleop', 'Pure Agent'], patch_artist=True,
            boxprops=dict(facecolor=box_colors), medianprops=dict(color='black'))

for box, color in zip(plt.boxplot([passthrough_execution_times, teleop_execution_times, pure_agent_execution_times],
                                  labels=['Passthrough', 'Teleop', 'Pure Agent'], patch_artist=True)['boxes'], box_colors):
    box.set_facecolor(color)

plt.ylabel('Execution Time (seconds)')
plt.title('Box Plot of Execution Time')
plt.grid(True, linestyle='--', alpha=0.7)

# Print mean and standard deviation in seconds
print('Passthrough Mean Execution Time:', passthrough_mean, 'seconds')
print('Passthrough Standard Deviation:', passthrough_std_dev, 'seconds')

print('\nTeleop Mean Execution Time:', teleop_mean, 'seconds')
print('Teleop Standard Deviation:', teleop_std_dev, 'seconds')

print('\nPure Agent Mean Execution Time:', pure_agent_mean, 'seconds')
print('Pure Agent Standard Deviation:', pure_agent_std_dev, 'seconds')

plt.show()
