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
    parser.add_argument('--data_path', help="Directory for data, e.g. './data/midi_files'.", required=True)
    parser.add_argument('--midi_dir', help="Name of data folder containing MIDI files, e.g. 'midis'.", required=True)
    args = parser.parse_args()
    return args

def main():
    """Main function: Converts selected MIDI files to pianoroll."""
    args = parse_arguments()

    midi_path = os.path.join(args.data_path, args.midi_dir)

    final_path = os.path.join(args.data_path, 'npz_files')
    if not os.path.exists(final_path):
        os.mkdir(final_path)

    for song in os.listdir(midi_path):
        midifile = os.path.join(midi_path, song)
        npz_path = os.path.join(final_path, song)

        if not os.path.exists(os.path.join(npz_path, '.npz')):
            try:
                parsed = pypianoroll.parse(midifile)
                pypianoroll.save(npz_path, parsed)
            except (IOError, IndexError) as e:
                pass

if __name__ == "__main__":
    main()
