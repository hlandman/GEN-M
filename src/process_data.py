# MIT License
#
# Copyright (c) 2018 Hao-Wen Dong and Wen-Yi Hsiao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Load and save an array to shared memory."""
import argparse
import os.path
import sys

import numpy as np
import SharedArray as sa


def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="Path to the data file.")
    parser.add_argument(
        "--name",
        help="File name to save in SharedArray. Defaults to the original file name.",
    )
    parser.add_argument(
        "--prefix",
        help="Prefix to the file name to save in SharedArray. Only effective when "
        "`name` is not given.",
    )
    parser.add_argument(
        "--dtype", default="bool", help="Datatype of the array. Defaults to bool."
    )
    args = parser.parse_args()
    return args.filepath, args.name, args.prefix, args.dtype


def create_shared_array(name, shape, dtype):
    """Create shared array. Prompt if a file with the same name existed."""
    try:
        return sa.create(name, shape, dtype)
    except FileExistsError:
        response = ""
        while response.lower() not in ["y", "n", "yes", "no"]:
            response = input(
                "Existing array (also named " + name + ") was found. Replace it? (y/n) "
            )
        if response.lower() in ("n", "no"):
            sys.exit(0)
        sa.delete(name)
        return sa.create(name, shape, dtype)


def main():
    """Load and save an array to shared memory."""
    filepath, name, prefix, dtype = parse_arguments()

    if name is None:
        name = os.path.splitext(os.path.basename(filepath))[0]
        if prefix is not None:
            name = prefix + "_" + name

    print("Loading data from '{}'.".format(filepath))
    if filepath.endswith(".npy"):
        data = np.load(filepath)
        data = data.astype(dtype)
        sa_array = create_shared_array(name, data.shape, data.dtype)
        print("Saving data to shared memory...")
        np.copyto(sa_array, data)
    else:
        with np.load(filepath) as loaded:
            sa_array = create_shared_array(name, loaded["shape"], dtype)
            print("Saving data to shared memory...")
            sa_array[[x for x in loaded["nonzero"]]] = 1

    print(
        "Successfully saved: (name='{}', shape={}, dtype={})".format(
            name, sa_array.shape, sa_array.dtype
        )
    )


if __name__ == "__main__":
    main()
