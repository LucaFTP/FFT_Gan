import sys

def is_power_of_two(n):
    """Check if an input value n is a multiple of 2."""
    return n > 0 and (n & (n - 1)) == 0

def validate_end_size(end_size):
    """Checks id END_SIZE is a power of two and less than 512."""
    if not is_power_of_two(end_size) or end_size > 512:
        print("Error: END_SIZE must be a power of two and less than 512")
        sys.exit(1)