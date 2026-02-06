"""Data loading utilities to match audio files with transcripts from JSON files."""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional


def load_json_file(json_path: str) -> List[dict]:
    """Load JSON file and return list of records.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        List of dictionaries from JSON file
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_transcript_for_audio(
    audio_filename: str,
    json_files: List[Tuple[str, str]]
) -> Optional[str]:
    """Search for transcript matching an audio filename in JSON files.
    
    Args:
        audio_filename: Name of the audio file (e.g., "file.wav")
        json_files: List of tuples (json_path, field_name) where field_name is either
                   'recording_file' or 'audio_file_path'
        
    Returns:
        Transcript string if found, None otherwise
    """
    # Search through each JSON file
    for json_path, field_name in json_files:
        if not os.path.exists(json_path):
            continue
        
        try:
            records = load_json_file(json_path)
            
            for record in records:
                # Get the audio filename from the JSON record
                json_audio_field = record.get(field_name)
                transcript = record.get('transcript', '')
                
                if json_audio_field and transcript:
                    # Normalize filename (handle path separators, get just the basename)
                    json_audio_filename = os.path.basename(json_audio_field)
                    
                    # Match by filename (case-insensitive comparison)
                    if json_audio_filename.lower() == audio_filename.lower():
                        return transcript
        except Exception as e:
            print(f"Warning: Error reading {json_path}: {e}")
            continue
    
    return None


def match_audio_with_transcripts(
    audio_dir: str,
    json_files: List[Tuple[str, str]],
    audio_files_dir: str = "dataset/audio_files"
) -> List[Tuple[str, str]]:
    """Match audio files with their transcripts from JSON files.
    
    First iterates through audio files, then searches JSON files for matching transcripts.
    
    Args:
        audio_dir: Directory containing audio files
        json_files: List of tuples (json_path, field_name) where field_name is either
                   'recording_file' or 'audio_file_path'
        audio_files_dir: Base directory for audio files (default: "dataset/audio_files")
        
    Returns:
        List of tuples (audio_path, transcript) for matched pairs
    """
    audio_transcript_pairs = []
    audio_dir_path = Path(audio_dir)
    
    # First, get all audio files
    if not audio_dir_path.exists():
        print(f"Warning: Audio directory not found: {audio_dir}")
        return audio_transcript_pairs
    
    audio_files = (
        list(audio_dir_path.glob("*.wav")) + 
        list(audio_dir_path.glob("*.mp3")) + 
        list(audio_dir_path.glob("*.flac"))
    )
    
    print(f"Found {len(audio_files)} audio files")
    
    # For each audio file, search in JSON files for matching transcript
    matched_count = 0
    for audio_file in audio_files:
        audio_filename = audio_file.name
        
        # Search for transcript in JSON files
        transcript = find_transcript_for_audio(audio_filename, json_files)
        
        if transcript:
            audio_transcript_pairs.append((str(audio_file), transcript))
            matched_count += 1
        else:
            print(f"Warning: No transcript found for {audio_filename}")
    
    print(f"Matched {matched_count} audio files with transcripts out of {len(audio_files)} total")
    return audio_transcript_pairs


def load_dataset(
    audio_files_dir: str = "dataset/audio_files",
    json_info_dir: str = "dataset/info"
) -> List[Tuple[str, str]]:
    """Load complete dataset by matching audio files with transcripts.
    
    Args:
        audio_files_dir: Directory containing audio files
        json_info_dir: Directory containing JSON files with transcripts
        
    Returns:
        List of tuples (audio_path, transcript)
    """
    json_files = [
        (os.path.join(json_info_dir, "postgres_laravel_calls.json"), "recording_file"),
        (os.path.join(json_info_dir, "postgres_public_call_transcripts.json"), "audio_file_path")
    ]
    
    return match_audio_with_transcripts(audio_files_dir, json_files, audio_files_dir)

