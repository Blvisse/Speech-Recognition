# Speech-Recognition

## Introduction
The project is inspired by the use of speech recognition systems in everyday life . Speech recognition systems sych as Siri, Alexa and google assitant convert speaach into action and hence automate and make work easier. 

This speech recognition models are however limited to a select few languages creating  a huge langauge barrier. The project aims to integrate swahili langauge into a speech to text system that allows the user to echo in commands in swahili which will then be converted into text. The speach-to-text system will be integrated in a mobile app for food purchase. 

The project seeks to use deep learning models that is capable of transcribing speech to text, with the aim of making the model accurate and robust against background noise
## Data
The data collected and reference for the swahili dataset can be found below:
### PUBLICATION ON SWAHILI SPEECH & LM DATA
More details on the corpus and how it was collected can be found on the following publication (please cite this bibtex if you use this data)

 @InProceedings { gelas:hal-00954048,
  author = {Gelas, Hadrien and Besacier, Laurent and Pellegrino, Francois},
  title = {{D}evelopments of {S}wahili resources for an automatic speech recognition system},
  booktitle = {{SLTU} - {W}orkshop on {S}poken {L}anguage {T}echnologies for {U}nder-{R}esourced {L}anguages},
  year = {2012},
  address = {Cape-Town, Afrique Du Sud},
  abstract = {no abstract},
  x-international-audience = {yes},
  url = {http://hal.inria.fr/hal-00954048},
}

### SWAHILI SPEECH CORPUS
Directory: /data/train
Files: text (training transcription), wav.scp (file id and path), utt2spk (file id and audio id), spk2utt (audio id and file id), wav (.wav files). 
For more information about the format, please refer to Kaldi website http://kaldi-asr.org/doc/data_prep.html
Description: training data in Kaldi format about 10 hours. Note: The path of wav files in wav.scp have to be modified to point to the actual locatiion.  

Directory: /data/test
Files: text (test transcription), wav.scp (file id and path), utt2spk (file id and audio id), spk2utt (audio id and file id), wav (.wav files)
Description: testing data in Kaldi format about 1.8 hours. The audio files for testing has the format 
ID_16k-emission_swahili_TThTT_-_TThTT_tu_YYYYMMDD_part001Q, where ID is the audio id, T is the time, Y is the year, M is the month and D is the day of recording. The 
last character Q indicate the quality of the utterance. g is good, m is utterance with background music, n is utterance with noise, s is utterance
with overlap speech, l is very noise utterance. Note: The path of wav files in wav.scp have to be modified to point to the actual locatiion. 



### SWAHILI TEXT CORPUS
Directory: /LM/
Files: 00-LM_SWH-CORPUS.txt, 01-CLN4-TRN.txt.zip, 02-CLN4-DEV.txt and 03-CLN4-TST.txt, swahili.arpa.zip

N.B.: You need to unzip 01-CLN4-TRN.txt.zip and swahili.arpa.zip for 01-CLN4-TRN.txt and swahili.arpa 

# /00-LM_SWH-CORPUS.txt
Contains 28 M Words. Full text data grabbed from online newspaper and cleaned as much as it could
All files below are extracted from this file

# /01-CLN4-TRN.txt
Training data for LM

# /02-CLN4-DEV.txt
Dev data for LM

# /03-CLN4-TST.txt
Testing data for LM

# /swahili.arpa
A language model created using SRILM [2] using the text from the text in 01-CLN4-TRN.txt



## Installation

Main packages used 

``` pip install dvc ```

``` pip install flask ```

``` pip install mlflow ```

``` pip install librosa ```

### import data from dvc

``` dvc pull ```

### Start flask serve

``` streamlt app.py ```

### View mlflow log 

``` mlflow ui ```

## Folders
```.dvc ``` DVC configuration

``` data ``` contains data versioned through dvc and also all audio files

``` scripts ``` contains modular scripts that are used to achieve tasks

``` plots ``` results plots from modelling

``` notebooks ``` all notebooks used for the project

``` logs ``` all run logs

``` tests ``` travis cml tests

## Scripts

``` audio.py ``` - This contains all the audio preprocessing functions

``` metadata.py ``` -this genrates a meta_data.json file for audio access

``` datalog ``` -Logs dvc data to mlflow

## Notebooks
``` Preprocess.ipynb ``` This involves all data anlysis and preprocessing

``` ModelTrainig.ipynb ``` Notebook to train deep learning models
## Badges 
[![GitHub license](https://img.shields.io/github/license/Blvisse/Speech-Recognition?style=for-the-badge)](https://github.com/Blvisse/Speech-Recognition/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/Blvisse/Speech-Recognition?style=for-the-badge)](https://github.com/Blvisse/Speech-Recognition/issues)
