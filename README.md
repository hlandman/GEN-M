## Data

### Lakh MIDI Dataset

1. Download [LMD-matched](http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz) file from [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/).  

Unzip file to ./data/.

2. Create directory `./data/lmd_matched/results`  
Download the following and add to results folder:  
[match_scores.json](http://hog.ee.columbia.edu/craffel/lmd/match_scores.json)  
[md5_to_paths.json](http://hog.ee.columbia.edu/craffel/lmd/md5_to_paths.json)  

Download [lmd_matched_h5](http://hog.ee.columbia.edu/craffel/lmd/lmd_matched_h5.tar.gz), unzip to results folder. 

3. Run preprocessing file to filter songs with the 'tech' and 'elec' keywords, longer than 60 seconds, and store them as .npz pianoroll files in `./data/lmd_matched/results/final_midis`.
```bash
python lakh_preprocess.py --data_path ./data/lmd_matched --midi_dir lmd_matched --meta_dir lmd_matched_h5 --keyword_list tech elec --min_length 60
```

Run `parser.py` and `compile.py`, both adapted from [symbolic-musical-datasets](https://github.com/wayne391/symbolic-musical-datasets/tree/master/5-track-pianoroll).  
`parser.py` finds all .npz files in the current directory, parses each instrument, and saves in the root directory as `lmd_segments.npy`.
```bash
python parser.py
```

`compile.py` compiles the parsed files in the correct input shape - in our case (4, 48, 84, 5).
```bash
python compile.py
```
