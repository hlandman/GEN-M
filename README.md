## Data

### Download Lakh MIDI Dataset

1. MIDI Data Download
	* Download [LMD-matched](http://hog.ee.columbia.edu/craffel/lmd/lmd_matched.tar.gz) file from [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/).  
	* Unzip file to ./data/.

2. Metadata Download
	* Create directory `./data/lmd_matched/results`  
	* Download the following and add to results folder:  
		* [match_scores.json](http://hog.ee.columbia.edu/craffel/lmd/match_scores.json)  
		* [md5_to_paths.json](http://hog.ee.columbia.edu/craffel/lmd/md5_to_paths.json)  
	* Download [lmd_matched_h5](http://hog.ee.columbia.edu/craffel/lmd/lmd_matched_h5.tar.gz), unzip to results folder.

### Preprocess MIDI Files

1. Run `lakh_preprocess.py` to filter songs from LMD with the 'tech' and 'elec' keywords, longer than 60 seconds, and store them as .npz pianoroll files in `./data/lmd_matched/results/npz_files`.

```bash
python lakh_preprocess.py --data_path ./data/lmd_matched --midi_dir lmd_matched --meta_dir lmd_matched_h5 --keyword_list tech elec --min_length 60
```  

2. Run `midi_preprocess.py` to convert and store other MIDI files as pianorolls in `./data/midi_files/npz_files`.

```bash
python midi_preprocess.py --data_path ./data/midi_files --midi_dir midis
```  

3. Run `parser.py` and `compile.py`, both adapted from [symbolic-musical-datasets](https://github.com/wayne391/symbolic-musical-datasets/tree/master/5-track-pianoroll).  
	* `parser.py` finds all .npz files in the current directory, parses each instrument, and saves in the root directory as `lmd_segments.npy`.
	* `compile.py` compiles the parsed files in the correct input shape - in our case (4, 48, 84, 5) - and saves as `data/lmd_compiled.npy`.

```bash
python parser.py
```  
```bash
python compile.py
```  
