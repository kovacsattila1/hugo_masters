import os
import re

def create_next_run_folder(path):
    # Regular expression to match "run" followed by a number
    pattern = re.compile(r'^run(\d+)$')
    
    highest_number = -1
    
    # Check if the path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    
    # Iterate over the items in the directory
    for dir_name in os.listdir(path):
        # Check if the directory name matches the pattern
        match = pattern.match(dir_name)
        if match:
            # Extract the number and convert to integer
            number = int(match.group(1))
            # Update the highest number if necessary
            if number > highest_number:
                highest_number = number
    
    # Increment the highest number by one for the new folder name
    next_number = highest_number + 1
    new_folder_name = f"run{next_number}"
    new_folder_path = os.path.join(path, new_folder_name)
    
    # Create the new folder
    os.makedirs(new_folder_path)
    
    return new_folder_path

# Example usage
path = "/home/kovacs/Documents/disszertacio/hugo_python_control_coppeliasim_v4/results"
new_folder = create_next_run_folder(path)
print(f"New folder created: {new_folder}")