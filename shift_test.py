def shift_elements(lst, new_elements):
    """
    Shifts three new elements into a list of length 18, initially filled with zeros.
    
    Parameters:
    lst (list): The original list of length 18.
    new_elements (list): A list of three new elements to be added.
    
    Returns:
    list: The updated list after shifting in the new elements.
    """
    if len(lst) != 18:
        raise ValueError("The original list must be of length 18.")
    if len(new_elements) != 3:
        raise ValueError("The new elements list must contain exactly three elements.")
    
    # Remove the last three elements
    lst = lst[:-3]
    # Add the new elements to the front
    lst = new_elements + lst
    return lst

# Example usage:
original_list = [0] * 18

new_elements = [1, 2, 3]
updated_list = shift_elements(original_list, new_elements)
print(updated_list)

new_elements = [1, 2, 3]
updated_list = shift_elements(updated_list, new_elements)
print(updated_list)

new_elements = [1, 2, 3]
updated_list = shift_elements(updated_list, new_elements)
print(updated_list)

new_elements = [1, 2, 3]
updated_list = shift_elements(updated_list, new_elements)
print(updated_list)