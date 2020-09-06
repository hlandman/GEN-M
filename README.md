## Data

### Lakh MIDI Dataset

Download LMD-matched file Lakh MIDI Dataset (https://colinraffel.com/projects/lmd/): 
http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz

Unzip to ./data/

Create directory ./data/lmd_matched/results
Download the following and add to results folder:
match_scores.json: http://hog.ee.columbia.edu/craffel/lmd/match_scores.json
md5_to_paths.json: http://hog.ee.columbia.edu/craffel/lmd/md5_to_paths.json
lmd_matched_h5 (unzip and add to folder): http://hog.ee.columbia.edu/craffel/lmd/lmd_matched_h5.tar.gz

Run preprocessing file to filter songs with the 'tech' and 'elec' keywords, longer than 60 seconds.
```bash
python lakh_preprocess.py --data_path ./data/lmd_matched --midi_dir lmd_matched --meta_dir lmd_matched_h5 /
--keyword_list tech elec --min_length 60
```