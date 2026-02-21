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


def build_transcript_lookup(json_files: List[Tuple[str, str]]) -> dict:
    """Build a dictionary mapping audio filenames to transcripts.

    This creates an O(1) lookup table for efficient transcript retrieval,
    avoiding the need for nested loops during matching.

    Args:
        json_files: List of tuples (json_path, field_name) where field_name
                   is the key containing the audio filename in each record

    Returns:
        Dictionary with lowercase filename as key and transcript as value
    """
    transcript_lookup = {}

    for json_path, field_name in json_files:
        if not os.path.exists(json_path):
            print(f"Warning: JSON file not found: {json_path}")
            continue

        try:
            records = load_json_file(json_path)
            for record in records:
                json_audio_field = record.get(field_name)
                transcript = record.get('transcript', '')

                if json_audio_field and transcript:
                    # Normalize filename and use lowercase for case-insensitive lookup
                    audio_filename = os.path.basename(json_audio_field)
                    lookup_key = audio_filename.lower()
                    transcript_lookup[lookup_key] = transcript
        except Exception as e:
            print(f"Warning: Error reading {json_path}: {e}")
            continue

    return transcript_lookup


def match_audio_with_transcripts(
    audio_dir: str,
    json_files: List[Tuple[str, str]],
    audio_files_dir: str = "dataset/audio_files"
) -> List[Tuple[str, str]]:
    """Match audio files with their transcripts from JSON files.

    Uses a dictionary-based O(1) lookup for efficient matching, building the
    lookup table once and then matching all audio files against it.

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

    # Build lookup dictionary from JSON files (O(J*R) where J=JSON files, R=records)
    transcript_lookup = build_transcript_lookup(json_files)
    print(f"Loaded {len(transcript_lookup)} transcript entries from JSON files")

    # Match audio files using O(1) dictionary lookup
    matched_count = 0
    for audio_file in audio_files:
        audio_filename = audio_file.name.lower()  # Case-insensitive lookup

        # O(1) lookup instead of O(J*R) search
        transcript = transcript_lookup.get(audio_filename)

        if transcript:
            audio_transcript_pairs.append((str(audio_file), transcript))
            matched_count += 1
        else:
            print(f"Warning: No transcript found for {audio_file.name}")

    print(f"Matched {matched_count} audio files with transcripts out of {len(audio_files)} total")
    return audio_transcript_pairs


def load_test_dataset(
    test_audio_dir: str = "dataset/test_audio"
) -> List[Tuple[str, str]]:
    """Load test dataset from test.json file in test_audio directory.
    
    This function reads transcripts directly from a test.json file located in the
    test_audio directory. The JSON file should be an array of objects, each with
    'audio_file_path' and 'transcript' fields.
    
    Args:
        test_audio_dir: Directory containing test audio files and test.json
        
    Returns:
        List of tuples (audio_path, transcript) for matched pairs
    """
    audio_transcript_pairs = []
    test_dir_path = Path(test_audio_dir)
    
    # Check if test directory exists
    if not test_dir_path.exists():
        print(f"Warning: Test audio directory not found: {test_audio_dir}")
        return audio_transcript_pairs
    
    # Look for test.json in the test_audio directory
    test_json_path = test_dir_path / "test.json"
    
    if not test_json_path.exists():
        print(f"Warning: test.json not found in {test_audio_dir}")
        return audio_transcript_pairs
    
    # Load test.json file
    try:
        test_records = load_json_file(str(test_json_path))
    except Exception as e:
        print(f"Error: Failed to load test.json: {e}")
        return audio_transcript_pairs
    
    if not isinstance(test_records, list):
        print(f"Error: test.json should contain an array of objects")
        return audio_transcript_pairs
    
    # Get all audio files in the directory (case-insensitive search)
    audio_files = {}
    for ext in ['*.wav', '*.WAV', '*.mp3', '*.MP3', '*.flac', '*.FLAC']:
        for audio_file in test_dir_path.glob(ext):
            # Store with lowercase filename for case-insensitive matching
            audio_files[audio_file.name.lower()] = audio_file
    
    print(f"Found {len(audio_files)} audio files in test directory")
    
    # Match audio files with transcripts from test.json
    matched_count = 0
    for record in test_records:
        if not isinstance(record, dict):
            continue
        
        audio_file_path = record.get('audio_file_path')
        transcript = record.get('transcript', '')
        
        if not audio_file_path or not transcript:
            continue
        
        # Get just the filename from the path (handle different path formats)
        json_audio_filename = os.path.basename(audio_file_path)
        
        # Try to find matching audio file (case-insensitive)
        audio_file_key = json_audio_filename.lower()
        
        if audio_file_key in audio_files:
            audio_file = audio_files[audio_file_key]
            audio_transcript_pairs.append((str(audio_file), transcript))
            matched_count += 1
        else:
            print(f"Warning: Audio file not found for {json_audio_filename} (from test.json)")
    
    print(f"Matched {matched_count} audio files with transcripts from test.json out of {len(test_records)} records")
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

