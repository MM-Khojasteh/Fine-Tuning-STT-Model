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


def match_audio_with_transcripts(
    audio_dir: str,
    json_files: List[Tuple[str, str]],
    audio_files_dir: str = "dataset/audio_files"
) -> List[Tuple[str, str]]:
    """Match audio files with their transcripts from JSON files.
    
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
    
    # Create a mapping from audio filename to transcript
    transcript_map = {}
    
    # Process each JSON file
    for json_path, field_name in json_files:
        if not os.path.exists(json_path):
            print(f"Warning: JSON file not found: {json_path}")
            continue
            
        records = load_json_file(json_path)
        
        for record in records:
            audio_filename = record.get(field_name)
            transcript = record.get('transcript', '')
            
            if audio_filename and transcript:
                # Normalize filename (handle path separators)
                audio_filename = os.path.basename(audio_filename)
                transcript_map[audio_filename] = transcript
    
    # Match audio files with transcripts
    if audio_dir_path.exists():
        audio_files = list(audio_dir_path.glob("*.wav")) + list(audio_dir_path.glob("*.mp3")) + list(audio_dir_path.glob("*.flac"))
        
        for audio_file in audio_files:
            audio_filename = audio_file.name
            if audio_filename in transcript_map:
                transcript = transcript_map[audio_filename]
                audio_transcript_pairs.append((str(audio_file), transcript))
            else:
                print(f"Warning: No transcript found for {audio_filename}")
    else:
        print(f"Warning: Audio directory not found: {audio_dir}")
    
    print(f"Found {len(audio_transcript_pairs)} audio-transcript pairs")
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

