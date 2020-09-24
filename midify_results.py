import os
import sys

from pypianoroll import Multitrack

def main():

    if len(sys.argv) < 2:
        print("Please include experiment directory path.")
        return 1
    elif not os.path.exists(sys.argv[1]):
        print("Experiment path does not exist.")
        return 1
    else:
        exp_path = sys.argv[1]

    # for t in ['fake_x_bernoulli_sampling', 'fake_x_hard_thresholding']:
    #     for i in range(10):
    #         result_path = os.path.join(exp_path, 'results/inference/pianorolls/{}/{}_{}.npz'.format(t, t, i))
    #         m = Multitrack(result_path)
    #         m.write(os.path.join(exp_path, 'results/inference/pianorolls/{}/result_{}.mid'.format(t, i)))
    #         print(t, i)

    method = 'fake_x_hard_thresholding'
    for i in range(10):
        result_path = os.path.join(exp_path, 'results/inference/pianorolls/{}/{}_{}.npz'.format(method, method, i))
        m = Multitrack(result_path)
        m.write(os.path.join(exp_path, 'results/inference/pianorolls/{}/result_{}.mid'.format(method, i)))
        print(method, i)

if __name__ == "__main__":
    main()
