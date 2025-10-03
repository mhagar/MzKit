import re

def natural_sort_key(text: str) -> list:
    """
    Generate a sort key for natural sorting

    Splits text into alternating string and number parts. Numbers are
    converted to integers for proper numeric sorting
    """
    def convert(part):
        if part.isdigit():
            return int(part)
        return part.lower()

    # Split on digits while keeping the digits
    parts = re.split(r'(\d+)', text)
    return [convert(part) for part in parts]
