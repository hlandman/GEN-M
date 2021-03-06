# GenM  

In this project we generate multi-track electronic music from scratch through deep learning. Our model is based on a configuration of the convolutional [Wasserstein GAN](https://arxiv.org/abs/1701.07875) developed by [MuseGAN](https://github.com/salu133445/musegan).   

We train our model on a subset of the [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/) that includes only tracks tagged as Electronic or Techno music.  

You can check out our [presentation](https://docs.google.com/presentation/d/1d_FcqL5FtC0PLgVvGZc5gGBLaW8jWxK2Io2w1n7CdGI/edit?usp=sharing) for information about our process,  experiments and results.  

We hope that musicians can use tracks generated by this model as a framework to create rich and intricate electronic music. You can check out a [sample audio snippet](https://youtu.be/p-b5NH49Bs8) where the model output's instruments are changed and the layers are looped and/or staggered.  

# Running Inference from Pre-trained Model  

Download the GenM Colab [Notebook](https://drive.google.com/file/d/12kX6oUFl9n8Skoi7283atH4vAiAT3Kzl/view?usp=sharing) and follow included instructions.  

# Training New Model  

## Hardware  

The full model was trained using an NVIDIA GTX 1070 ti 16gb GPU and took ~6 hours to train for 10,000 steps.  

## Environment

1. Clone repo  
```bash
git clone https://github.com/hlandman/GEN-M.git
```  

2. Install requirements  
```bash
pip install -r requirements.txt
```  
Requirements include tensorflow 1.10 - this presents potential environment challenges.

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
python preprocess/lakh_preprocess.py --data_path ./data/lmd_matched --midi_dir lmd_matched \
--meta_dir lmd_matched_h5 --keyword_list tech elec --min_length 60
```  

2. Run `midi_preprocess.py` to convert and store other MIDI files as pianorolls in `./data/midi_files/npz_files`.

```bash
python preprocess/midi_preprocess.py --data_path ./data/midi_files --midi_dir midis
```  

3. Run `parser.py` and `compile.py`, both adapted from [symbolic-musical-datasets](https://github.com/wayne391/symbolic-musical-datasets/tree/master/5-track-pianoroll).  
	* `parser.py` finds all .npz files in the current directory, parses each instrument, and saves in the root directory as `data/midi_segments.npy`.
	* `compile.py` compiles the parsed files in the correct input shape - in our case (4, 48, 84, 5) - and saves as `data/midi_compiled.npy`.

```bash
python preprocess/parser.py
```  
```bash
python preprocess/compile.py
```  

## Train Model

### Create New "Experiment"

1. If it doesn't exist yet, create a directory for experiments.  

```bash
mkdir ./exp
```  

2. Create a new experiment and copy the default config and params yaml files.  

```bash
mkdir ./exp/lmd_tech_elec_01 # Or custom experiment name
cp ./default_params.yaml ./exp/lmd_tech_elec_01/params.yaml
cp ./default_config.yaml ./exp/lmd_tech_elec_01/config.yaml
```  

3. Add a note describing the experiment.  

```bash
echo "Train on 3690 Tech/Elec songs from LMD - 10,000 steps." > ./exp/lmd_tech_elec_01/exp_note.txt
```  

4. Manually edit `./exp/<exp_name>/config.yaml` according to experiment specs.

Important values we used:  
```
# Experiment  
save_checkpoint_steps: 50  

# Data  
data_source: 'npy'  
data_filename: './data/midi_compiled.npy'  

# Training  
steps: 10000  
batch_size: 32  

# Sampling  
midi:  
	tempo: 120  
```  

### Run Model Training  

```bash
python ./src/train.py --exp_dir ./exp/lmd_tech_elec_01 --params \
./exp/lmd_tech_elec_01/params.yaml --config ./exp/lmd_tech_elec_01/config.yaml --gpu 0
 ```  

## Inference

1. Run inference

```bash
python ./src/inference.py --checkpoint_dir ./exp/lmd_tech_elec_01/model \
--result_dir ./exp/lmd_tech_elec_01/results/inference --params ./exp/lmd_tech_elec_01/params.yaml \
  --config ./exp/lmd_tech_elec_01/config.yaml --runs 10 --gpu 0
```  

2. Convert `.npz` results to midi  
```bash
python midify_results.py ./exp/lmd_tech_elec_01/
```  
Resulting MIDI files are stored in `./exp/<exp_name>/results/inference/pianorolls`.  

## Evaluation  

Run `run_eval.py <exp_name>` to see metrics for resulting midis.  
```bash
python run_eval.py lmd_tech_elec_01
```  
