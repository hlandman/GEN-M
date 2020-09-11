import os
import json
import h5py
import argparse
import numpy as np
import pretty_midi
import pypianoroll

def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help="Directory for data, e.g. './data/lmd_matched'.", required=True)
    parser.add_argument('--midi_dir', help="Name of data folder containing MIDI files, e.g. 'lmd_matched'.", required=True)
    parser.add_argument('--meta_dir', help="Name of metadata folder, e.g. 'lmd_matched_h5'.", required=True)
    parser.add_argument('--keyword_list', help="List of genre keywords to filter for in artist terms.", nargs='+', required=True)
    parser.add_argument('--min_length', help="Minimum song length in seconds.", required=True)
    args = parser.parse_args()
    return args

def get_scores():
    """Retrieve match_scores.json, which matches Lakh MIDI files to
    corresponding 'MSD ID' tags for songs in the Million Song Dataset."""
    args = parse_arguments()
    SCORE_FILE = os.path.join(args.data_path, 'results/match_scores.json')

    with open(SCORE_FILE, 'r') as f:
        scores = json.load(f)

    return scores

def msd_id_to_dirs(msd_id):
    """Given an MSD ID (based on "scores" file keys), generate the path prefix to the MIDI file.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

def msd_id_to_h5(msd_id):
    """Given an MSD ID, return the path to the corresponding h5 with song metadata."""
    args = parse_arguments()
    METADATA_PATH = os.path.join(args.data_path, 'results', args.meta_dir)

    return os.path.join(METADATA_PATH, msd_id_to_dirs(msd_id) + '.h5')

def midi_path(msd_id):
    """Obtain MD5 and return full path to MIDI file.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678/0abcdef1g2h34i5_..._.mid"""
    args = parse_arguments()
    scores = get_scores()
    md5 = list(scores[msd_id].keys())[0]
    return os.path.join(args.data_path, args.midi_dir, msd_id_to_dirs(msd_id), md5 + '.mid')

def fliter_genres():
    """Get list of song ids whose genre tags contain keywords."""
    args = parse_arguments()

    song_list = []

    scores = get_scores()
    msd_id_list = [i for i in scores.keys()]
    # msd_id_list = list(get_scores())[:50]

    # Iterate through song IDs
    for msd_id in msd_id_list:
        # Read metadata for song
        metadata = h5py.File(msd_id_to_h5(msd_id), "r")

        # Iterate through top-5 "artist_terms" (i.e. genre tags) and check if they contain keywords.
        has_key = False
        for term in [str(i, 'utf-8') for i in metadata['metadata']['artist_terms'][:5]]:
            if np.any([i in term for i in args.keyword_list]):
                has_key = True

        if has_key:
            song_list.append(msd_id)

    return song_list

def get_long_songs():
    """Get songs longer than min length that are in 4/4 time."""
    args = parse_arguments()

    long_songs = []

    for song in fliter_genres():
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path(song))
            if midi_data.get_end_time() >= int(args.min_length):
                if midi_data.time_signature_changes[0].numerator == 4:
                    long_songs.append(song)
        except:
            pass

    return long_songs

def main():
    """Main function: Converts selected MIDI files to pianoroll."""
    args = parse_arguments()

    songs = get_long_songs()

    final_path = os.path.join(args.data_path, 'results', 'npz_files')
    if not os.path.exists(final_path):
        os.mkdir(final_path)

    for song in songs:
        midifile = midi_path(song)
        npz_path = os.path.join(final_path, song)

        if not os.path.exists(os.path.join(npz_path, '.npz')):
            try:
                parsed = pypianoroll.parse(midifile)
                pypianoroll.save(npz_path, parsed)
            except (IOError, IndexError) as e:
                pass

if __name__ == "__main__":
    main()
