import muspy
import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Please include experiment name.")
        return 1
    elif not os.path.exists(os.path.join('exp', sys.argv[1])):
        print("Experiment does not exist.")
        return 1
    else:
        path = os.path.join('exp', sys.argv[1], 'results', 'inference',
                            'pianorolls', 'fake_x_hard_thresholding')

    n = 0

    ## Pitch
    pitch_range = 0
    number_pitches_used = 0
    number_pitch_classes_used = 0
    pitch_entropy = 0
    pitch_class_entropy = 0
    polyphony = 0
    polyphony_rate = 0
    scale_consistency = 0

    ## Rhythm
    empty_beat_rate = 0
    empty_measure_rate = 0
    drum_pattern_consistency = 0
    drum_in_pattern_rate = 0
    groove_consistency = 0

    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.mid'):
                music = muspy.read_midi(os.path.join(dirpath, filename), backend="mido")

                n += 1
                ## Pitch
                pitch_range += muspy.pitch_range(music)
                number_pitches_used += muspy.n_pitches_used(music)
                number_pitch_classes_used += muspy.n_pitch_classes_used(music)
                pitch_entropy += muspy.pitch_entropy(music)
                pitch_class_entropy += muspy.pitch_class_entropy(music)
                polyphony += muspy.polyphony(music)
                polyphony_rate += muspy.polyphony_rate(music)
                scale_consistency += muspy.scale_consistency(music)

                ## Rhythm
                empty_beat_rate += muspy.empty_beat_rate(music)
                empty_measure_rate += muspy.empty_measure_rate(music, 16)
                drum_pattern_consistency += muspy.drum_pattern_consistency(music)
                groove_consistency += muspy.groove_consistency(music, 16)

        print("Number of files analyzed: ", n)
        print("\n#######################")
        print("Pitch-Related Metrics")
        print("#######################\n")

        print("Pitch Range: ", round(pitch_range / n))
        print("Number of Pitches Used: ", round(number_pitches_used / n))
        print("Number of Pitch Classes Used: ", round(number_pitch_classes_used / n))
        print("Pitch Entropy: ", round(pitch_entropy / n, 2))
        print("Pitch Class Entropy: ", round(pitch_class_entropy / n, 2))
        print("Polyphony: ", round(polyphony / n, 2))
        print("Polyphony Rate: ", round(polyphony_rate / n, 2))
        print("Scale Consistency: ", round(scale_consistency / n, 2))

        print("\n#######################")
        print("Rhythm-Related Metrics")
        print("#######################\n")

        print("Empty Beat Rate: ", round(empty_beat_rate / n, 2))
        print("Empty Measure Rate: ", round(empty_measure_rate / n, 2))
        print("Drum Pattern Consistency: ", round(drum_pattern_consistency / n, 2))
        print("Groove Consistency: ", round(groove_consistency / n, 2))

if __name__ == '__main__':
    main()
