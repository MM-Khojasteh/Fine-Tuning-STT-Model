"""Utility functions for the fine-tuning project."""

def format_timestamp(seconds):
    """Format seconds into MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string in MM:SS format
    """
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:02}:{secs:02}"

