import numpy as np

def map_to_discrete_range(value, input_min, input_max, output_min, output_max):
    # Clip the value to the input range
    value = np.clip(value, input_min, input_max)
    
    # Normalize the value to a 0-1 range
    normalized_value = (value - input_min) / (input_max - input_min)
    
    # Scale to the output range and round to the nearest integer
    discrete_value = np.round(normalized_value * (output_max - output_min) + output_min).astype(int)
    
    return discrete_value

# Example usage:
value = 0
input_min, input_max = -1, 1
output_min, output_max = 0, 180
mapped_value = map_to_discrete_range(value, input_min, input_max, output_min, output_max)
print(mapped_value)  # Output will be 3


value = -0.5
input_min, input_max = -1, 1
output_min, output_max = 0, 180
mapped_value = map_to_discrete_range(value, input_min, input_max, output_min, output_max)
print(mapped_value)  # Output will be 3


value = 0.5
input_min, input_max = -1, 1
output_min, output_max = 0, 180
mapped_value = map_to_discrete_range(value, input_min, input_max, output_min, output_max)
print(mapped_value)  # Output will be 3
