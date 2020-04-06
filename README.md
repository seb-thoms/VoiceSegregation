# VoiceSegregation

The original repo is [here](https://github.com/taylorlu/Speaker-Diarization).

The original repo has a great speaker diarization model using vgg and uis-rnn. Hats off to [taylorlu](https://github.com/taylorlu)

Made some tweaks in the final file SpeakerDiarization.py to actually have the transciption.

Speech to text is done by IBM's Watson API.

To run, follow the below steps

1) Download the api credentials json file for IBM Watson's speech to text and save it as 'api.json' in the same directory of project.

2) Create 'input.json' file containing the above values
  {
    "audio_file_path": "audio.wav",
    "create_output_directory": 1
  }
  First key specifies the path of audio file to transcribe. 
  I added a functionality to store the seperate audio clips for each speaker and save it in a directory. For this set second key to 1 else 0
  
 And that's it. Run the file and you'll have the transcribed output.
 
 May the force be with you all 🖖 !!

