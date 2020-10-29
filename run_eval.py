import muspy
import os
import glob
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

    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            n = 0

            ## Pitch
            pitch_range = 0
            number_pitches_used = 0
            number_pitch_classes_used = 0
            polyphony = 0
            scale_consistency = 0

            ## Rhythm
            empty_beat_rate = 0
            empty_measure_rate = 0
            drum_pattern_consistency = 0
            groove_consistency = 0

            if filename.endswith('.mid'):
                music = muspy.read_midi(os.path.join(dirpath, filename), backend = "mido")

                n += 1
                ## Pitch
                pitch_range += muspy.pitch_range(music)
                number_pitches_used += muspy.n_pitches_used(music)
                number_pitch_classes_used += muspy.n_pitch_classes_used(music)
                polyphony += muspy.polyphony(music)
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

        print("Pitch Range: ", pitch_range/n)
        print("Number of Pitches Used: ", number_pitches_used/n)
        print("Number of Pitch Classes Used: ", number_pitch_classes_used/n)
        print("Polyphony: ", polyphony/n)
        print("Scale Consistency: ", scale_consistency/n)

        print("\n#######################")
        print("Rhythm-Related Metrics")
        print("#######################\n")

        print("Empty Beat Rate: ", empty_beat_rate/n)
        print("Empty Measure Rate: ", empty_measure_rate/n)
        print("Drum Pattern Consistency: ", drum_pattern_consistency/n)
        print("Groove Consistency: ", groove_consistency/n)

if __name__ == '__main__':
    main()
