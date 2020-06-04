
"""A demo script showing how to DIARIZATION ON WAV USING UIS-RNN."""

import numpy as np
import uisrnn
import librosa
import sys
import os
import shutil
import json
from datetime import datetime
from pydub import AudioSegment
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
sys.path.append('ghostvlad')
sys.path.append('visualization')
import toolkits
import model as spkModel
from viewer import PlotDiar

# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default=r'ghostvlad/pretrained/weights.h5', type=str)
parser.add_argument('--data_path', default='4persons', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

# GLOBAL VARIABLES
global args
args = parser.parse_args()


SAVED_MODEL_NAME = 'pretrained/saved_model(500).uisrnn_benchmark'

audio_input = None
final_transcript = {}
dir_name = ''


def load_audio(audio_file_path):
    global audio_input
    audio_input = AudioSegment.from_wav(audio_file_path)


def append2dict(speakerSlice, spk_period):
    key = list(spk_period.keys())[0]
    value = list(spk_period.values())[0]
    timeDict = {}
    timeDict['start'] = int(value[0]+0.5)
    timeDict['stop'] = int(value[1]+0.5)
    if(key in speakerSlice):
        speakerSlice[key].append(timeDict)
    else:
        speakerSlice[key] = [timeDict]

    return speakerSlice


def arrangeResult(labels, time_spec_rate): # {'1': [{'start':10, 'stop':20}, {'start':30, 'stop':40}], '2': [{'start':90, 'stop':100}]}
    lastLabel = labels[0]
    speakerSlice = {}
    j = 0
    for i,label in enumerate(labels):
        if(label==lastLabel):
            continue
        speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate*j,time_spec_rate*i)})
        j = i
        lastLabel = label
    speakerSlice = append2dict(speakerSlice, {lastLabel: (time_spec_rate*j,time_spec_rate*(len(labels)))})
    return speakerSlice


def genMap(intervals):  # interval slices to maptable
    slicelen = [sliced[1]-sliced[0] for sliced in intervals.tolist()]
    mapTable = {}  # vad erased time to origin time, only split points
    idx = 0
    for i, sliced in enumerate(intervals.tolist()):
        mapTable[idx] = sliced[0]
        idx += slicelen[i]
    mapTable[sum(slicelen)] = intervals[-1,-1]

    keys = [k for k,_ in mapTable.items()]
    keys.sort()
    return mapTable, keys


def fmtTime(timeInMillisecond):
    millisecond = timeInMillisecond%1000
    minute = timeInMillisecond//1000//60
    second = (timeInMillisecond-minute*60*1000)//1000
    time = '{}:{:02d}.{}'.format(minute, second, millisecond)
    return time


def load_wav(vid_path, sr):
    wav, _ = librosa.load(vid_path, sr=sr)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
      wav_output.extend(wav[sliced[0]:sliced[1]])
    return np.array(wav_output), (intervals/sr*1000).astype(int)


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


# 0s        1s        2s                  4s                  6s
# |-------------------|-------------------|-------------------|
# |-------------------|
#           |-------------------|
#                     |-------------------|
#                               |-------------------|
def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, embedding_per_second=0.5, overlap_rate=0.5):
    wav, intervals = load_wav(path, sr=sr)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T

    spec_len = sr/hop_length/embedding_per_second
    spec_hop_len = spec_len*(1-overlap_rate)

    cur_slide = 0.0
    utterances_spec = []

    while(True):  # slide window.
        if(cur_slide + spec_len > time):
            break
        spec_mag = mag_T[:, int(cur_slide+0.5) : int(cur_slide+spec_len+0.5)]
        
        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

        cur_slide += spec_hop_len

    return utterances_spec, intervals


def speech2text(audio_file_path):
    """
    This function takes an audio clip in wav format and returns the text using the Watson speech API by IBM
    Args:
        audio_file_path (str) : File path of the audio file
    Returns:
        text (str) : Corresponding text
    """
    try:
        api_json = open('api.json')
        api_json = json.load(api_json)
        authenticator = IAMAuthenticator(api_json['apikey'])
        speech_to_text = SpeechToTextV1(authenticator=authenticator)

        speech_to_text.set_service_url(api_json['url'])

        speech_to_text.set_disable_ssl_verification(True)

        with open(audio_file_path, 'rb') as audio_file:
            speech_recognition_results = speech_to_text.recognize(
                audio=audio_file,
                content_type='audio/wav'
            ).get_result()
        text = ''
        results = speech_recognition_results['results']
        for item in results:
            transcipt = item['alternatives'][0]['transcript']
            text = text + ' ' + transcipt
        return text
    except Exception as exp:
        print(f"Failed in speech2text with error {exp}")


def get_transcript(speaker, start_time, end_time):
    """
    This function takes a speaker, start time and end time of his segment in the audio clip and stores the
    transcript in final_transcript dictionary

    Args:
        speaker (str) : Label of the corresponding speaker (eg : 0, 1 , 2)
        start_time (int)  : starting time of audio segment in milliseconds
        end_time (int) : ending time of audio segment in milliseconds

    Returns:
        None
    """
    try:
        audio_file_name = '/Speaker' + speaker + '_' + str(start_time) + '_' + str(end_time)
        audio_file_name += '.wav'
        audio_clip = audio_input[start_time:end_time]
        audio_clip.export(dir_name + audio_file_name, format='wav')
        transcript = speech2text(dir_name + audio_file_name)
        os.remove(dir_name + audio_file_name)

        if speaker in final_transcript.keys():
            final_transcript[speaker].append((start_time, end_time, transcript))
            speaker_audio = AudioSegment.from_wav(dir_name + '/Speaker' + speaker +'.wav')
            merged = speaker_audio + audio_clip
            merged.export(dir_name + '/Speaker' + speaker, format='wav' )
        else:
            final_transcript[speaker] = [(start_time, end_time, transcript)]
            speaker_file_name = '/Speaker' + speaker
            speaker_file_name += '.wav'
            audio_clip.export(dir_name + speaker_file_name, format='wav')
    except Exception as exp:
        print(f"Failed in get_transcript with error {exp}")


def print_transcipt():
    """
    This function sorts the final_transcript dictionary according to the time stamps and prints
    the final transcript

    Returns:
        ordered_transcript (list) : Sorted transcripts are stored as (speaker, start time, end time, speech)
    """
    ordered_transcript = []
    index = 0
    for speaker in final_transcript:
        for text in final_transcript[speaker]:
            start = text[0]
            end = text[1]
            speech = text[2]
            for i in range(len(ordered_transcript)):
                if i == len(ordered_transcript)-1:
                    if start > ordered_transcript[i][1]:
                        index = i+1
                        break
                else:
                    if ordered_transcript[i][1] < start < ordered_transcript[i + 1][1]:
                        index = i+1
                        break
            ordered_transcript = ordered_transcript[:index] + [(speaker, start, end, speech)] + ordered_transcript[index:]
    return ordered_transcript


def main(wav_path, embedding_per_second=1.0, overlap_rate=0.5, retain_audio_clip=False):

    # gpu configuration
    toolkits.initialize_GPU(args)

    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True,
              }

    network_eval = spkModel.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval', args=args)
    network_eval.load_weights(args.resume, by_name=True)


    model_args, _, inference_args = uisrnn.parse_arguments()
    model_args.observation_dim = 512
    uisrnnModel = uisrnn.UISRNN(model_args)
    uisrnnModel.load(SAVED_MODEL_NAME)

    specs, intervals = load_data(wav_path, embedding_per_second=embedding_per_second, overlap_rate=overlap_rate)
    mapTable, keys = genMap(intervals)

    feats = []
    for spec in specs:
        spec = np.expand_dims(np.expand_dims(spec, 0), -1)
        v = network_eval.predict(spec)
        feats += [v]

    feats = np.array(feats)[:,0,:].astype(float)  # [splits, embedding dim]
    predicted_label = uisrnnModel.predict(feats, inference_args)

    time_spec_rate = 1000*(1.0/embedding_per_second)*(1.0-overlap_rate) # speaker embedding every ?ms
    center_duration = int(1000*(1.0/embedding_per_second)//2)
    speakerSlice = arrangeResult(predicted_label, time_spec_rate)

    for spk,timeDicts in speakerSlice.items():    # time map to orgin wav(contains mute)
        for tid,timeDict in enumerate(timeDicts):
            s = 0
            e = 0
            for i,key in enumerate(keys):
                if(s!=0 and e!=0):
                    break
                if(s==0 and key>timeDict['start']):
                    offset = timeDict['start'] - keys[i-1]
                    s = mapTable[keys[i-1]] + offset
                if(e==0 and key>timeDict['stop']):
                    offset = timeDict['stop'] - keys[i-1]
                    e = mapTable[keys[i-1]] + offset

            speakerSlice[spk][tid]['start'] = s
            speakerSlice[spk][tid]['stop'] = e

    for spk, timeDicts in speakerSlice.items():
        for timeDict in timeDicts:
            s = timeDict['start']
            e = timeDict['stop']
            get_transcript(str(spk), s, e)

    result = print_transcipt()
    for item in result:
        start = fmtTime(item[1])
        end = fmtTime(item[2])
        print(f"{start} ==> {end}: [Speaker : {item[0]}] : {item[3]}")

    if not retain_audio_clip:
        shutil.rmtree(dir_name)
    else:
        print(f'Audio files of transcriptions can be found in {dir_name} folder')

    p = PlotDiar(map=speakerSlice, wav=wav_path, gui=True, size=(25, 6))
    p.draw()
    p.plot.show()

    return result


def server_entry_point(inputs):
    """
    Entry point for server call

    Args:
        inputs (dict) : Should contain keys, username(str), audio_file(base64 encoded string) and create_output_directory(0 or 1)

    Returns:
        The final transcription in a dict format
    """
    import base64
    global dir_name
    timestamp = str(datetime.now())
    timestamp = timestamp[11:19].replace(':', '')
    decode_string = base64.b64decode(inputs["audio_file"])
    audio_file_name = "Audio_" + timestamp + ".wav"
    file = open(audio_file_name, "wb")
    file.write(decode_string)
    file.close()
    dir_name = inputs["username"] + timestamp
    os.mkdir(dir_name)
    load_audio(audio_file_name)
    result = main(audio_file_name,
         embedding_per_second=1.2,
         overlap_rate=0.4)
    print("server", result)
    return {"finalTranscipt": result}


if __name__ == '__main__':
    input_json = open('input.json')
    inputs = json.load(input_json)
    
    # Time stamp is used to create an output directory of separate audio files
    timestamp = str(datetime.now())
    timestamp = timestamp[11:19].replace(':', '')
    dir_name = inputs["audio_file_path"].split('/').pop() + timestamp
    os.mkdir(dir_name)
    audio_file_path = inputs["audio_file_path"]

    # To handle incoming mp3 files. Converts mp3 to wav
    if audio_file_path.endswith(".mp3"):
        audio = AudioSegment.from_mp3(audio_file_path)
        audio_file_path = audio_file_path[:-4]+".wav"
        audio.export(audio_file_path, format="wav")
    
    # audio file is loaded
    load_audio(audio_file_path)

    # Voice Separation function call
    main(audio_file_path,
         embedding_per_second=1.2,
         overlap_rate=0.8,
         retain_audio_clip=True if inputs['create_output_directory'] == 1 else False)


# InsecureRequestWarning: Unverified HTTPS request is being made to host 'api.eu-gb.speech-to-text.watson.cloud.ibm.com'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
#   InsecureRequestWarning,

# Fix %hesitation

# Combine audio segemnt of same speakers into one and then save


