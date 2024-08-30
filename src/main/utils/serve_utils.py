def parse_temp(temp: float) -> float:
    """Handle invalid temperature values."""
    if temp > 1:
        return 1
    if temp < 0:
        return 0
    return temp