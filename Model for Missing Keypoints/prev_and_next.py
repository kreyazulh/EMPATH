def generate_new_list(sorted_numbers):
    new_list = []

    # Iterate through the sorted list
    i = 0
    while i < len(sorted_numbers):
        current_number = sorted_numbers[i]

        # Find the range for consecutive numbers
        start = current_number
        while i + 1 < len(sorted_numbers) and sorted_numbers[i + 1] == current_number + 1:
            current_number = sorted_numbers[i + 1]
            i += 1

        end = current_number

        # Add the range to the new list
        new_list.append([start - 1, end + 1])

        # Move to the next number in the sorted list
        i += 1

    return new_list
