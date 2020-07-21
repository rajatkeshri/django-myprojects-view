from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import json
import tensorflow as tf
import numpy as np
import librosa
import os

from os import path
from pydub import AudioSegment

#############################################################################
def convert_to_wav(src,dst):
    # convert wav to mp3
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")

#############################################################################
def index(request):
    context = {}
    return render(request,"sound_ml/index.html",context)

#############################################################################
def speech2text(request):
    context={}
    context["predicted_True"] = 0
    context["mappings"] = {}
    context["highest"] = -1
    prediction_dic = {}
    input_file_name = ""

    SAVED_MODEL_PATH = "sound_ml/static/sound_ml/speech2textmodel.h5"
    SAMPLES_TO_CONSIDER = 22050
    model = ""


    print("HELLO")

    mappings = ["Down","Go","Left","No","Off","On","Right","Stop","Up","Yes"]

    num_mfcc=13
    hop_length=512
    n_fft=2048

    #########################################################
    # get all test data file names
    test_data_path = "sound_ml/static/sound_ml/test_speech2text/"
    test_data_path_frontend = "static/sound_ml/test_speech2text/"
    test_files_paths = []
    file_filepath_dic = {}

    test_files = os.listdir(test_data_path)
    print(test_files)

    for i in test_files :
        test_files_paths.append(test_data_path_frontend + str(i))

    c = 0
    for i in test_files:
        i = i.split(".")[0]
        file_filepath_dic[i] = test_files_paths[c]
        c+=1

    context["test_files_dic"] = file_filepath_dic


    ########################################################


    try:
        try:
            myfile = request.FILES["myfile"]
            flag = 0
        except:
            flag = 1

        if flag == 0:
            if request.method == 'POST' and request.FILES['myfile']:
                myfile = request.FILES['myfile']
                fs = FileSystemStorage()
                filename = fs.save(myfile.name, myfile)
                uploaded_file_url = fs.url(filename)
                context['uploaded_file_url'] = uploaded_file_url[1:]

                uploaded_file_url = uploaded_file_url[1:]

                #split_path = uploaded_file_url.split("/")
                #convert_to_wav(uploaded_file_url,str(split_path[0])+str("/1.wav"))
                #os.remove(uploaded_file_url)
                #uploaded_file_url = str(split_path[0])+str("/1.wav")

                #load model
                model = tf.keras.models.load_model(SAVED_MODEL_PATH)
                print("hello")
                #model.summary()

                #load file
                signal, sample_rate = librosa.load(uploaded_file_url)

                #extract mfcc
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    # ensure consistency of the length of the signal
                    signal = signal[:SAMPLES_TO_CONSIDER]
                    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    MFCCs = MFCCs.T
                    print(MFCCs.shape)

                    #change dimensions
                    MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

                    #predict
                    predictions = model.predict(MFCCs)
                    predicted_index = np.argmax(predictions)
                    predicted_keyword = mappings[predicted_index]

                    predictions = [i * 100 for i in predictions]
                    predictions = list(predictions[0])

                    c=0
                    for i in mappings:
                        prediction_dic[i] = predictions[c]
                        c+=1

                    context["predicted_True"] = 1
                    context["mappings"] = prediction_dic
                    context["predicted_keyword"] = predicted_keyword
                    context["highest"] = predicted_index+1
                else:
                    context["predicted_True"] = 2
                    print("less than 1", len(signal))

                os.remove(uploaded_file_url)
        else:
            if request.method == 'POST':
                for i in request.POST:
                    if request.POST[i] == "on":
                        input_file_name = str(i)

                if input_file_name == "":
                    context["is_none"] = 1
                    return render(request,"sound_ml/speech2text.html",context)

                uploaded_file_url = test_data_path + input_file_name + ".wav"

                #load model
                model = tf.keras.models.load_model(SAVED_MODEL_PATH)
                print("hello")
                #model.summary()

                #load file
                signal, sample_rate = librosa.load(uploaded_file_url)

                #extract mfcc
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    # ensure consistency of the length of the signal
                    signal = signal[:SAMPLES_TO_CONSIDER]
                    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    MFCCs = MFCCs.T
                    print(MFCCs.shape)

                    #change dimensions
                    MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

                    #predict
                    predictions = model.predict(MFCCs)
                    predicted_index = np.argmax(predictions)
                    predicted_keyword = mappings[predicted_index]

                    predictions = [i * 100 for i in predictions]
                    predictions = list(predictions[0])

                    c=0
                    for i in mappings:
                        prediction_dic[i] = predictions[c]
                        c+=1

                    context["predicted_True"] = 1
                    context["mappings"] = prediction_dic
                    context["predicted_keyword"] = predicted_keyword
                    context["highest"] = predicted_index+1
                else:
                    context["predicted_True"] = 2
                    print("less than 1", len(signal))



    except:
        context["predicted_True"] = 2
        if request.method == 'POST' and request.FILES['myfile']:
            if flag == 0:
                os.remove(uploaded_file_url)

    tf.keras.backend.clear_session()
    return render(request,"sound_ml/speech2text.html",context)


#############################################################################
def musicgenre(request):
    context={}

    context["prediction"] = "none"
    input_file_name =""

    #Constants which depend on the model. If you train the model with different values,
    #need to change those values here too
    num_mfcc = 13
    n_fft=2048
    hop_length = 512
    sample_rate = 22050
    samples_per_track = sample_rate * 30
    num_segment = 10

    SAVED_MODEL_PATH = "sound_ml/static/sound_ml/model_RNN_LSTM.h5"

    classes = ["Blues","Classical","Country","Disco","Hiphop",
                "Jazz","Metal","Pop","Reggae","Rock"]

    class_predictions = []
    samples_per_segment = int(samples_per_track / num_segment)


    #########################################################
    # get all test data file names
    test_data_path = "sound_ml/static/sound_ml/test_musicgenre/"
    test_data_path_frontend = "static/sound_ml/test_musicgenre/"
    test_files_paths = []
    file_filepath_dic = {}

    test_files = os.listdir(test_data_path)
    print(test_files)

    for i in test_files :
        test_files_paths.append(test_data_path_frontend + str(i))

    c = 0
    for i in test_files:
        i = i.split(".")[0]
        file_filepath_dic[i] = test_files_paths[c]
        c+=1

    context["test_files_dic"] = file_filepath_dic


    ########################################################


    try:
        try:
            myfile = request.FILES["myfile"]
            flag = 0
        except:
            flag = 1

        if flag == 0:
            if request.method == 'POST' and request.FILES['myfile']:
                myfile = request.FILES['myfile']
                fs = FileSystemStorage()
                filename = fs.save(myfile.name, myfile)
                uploaded_file_url = fs.url(filename)
                context['uploaded_file_url'] = uploaded_file_url[1:]

                uploaded_file_url = uploaded_file_url[1:]
                song_path = str(uploaded_file_url)
                print(uploaded_file_url)

                if song_path.endswith('.mp3'):
                    path_to_save = "media/" + "m"+".wav"
                    convert_to_wav(song_path,path_to_save)
                    song_path = path_to_save
                else:
                    pass

                #split_path = uploaded_file_url.split("/")
                #convert_to_wav(uploaded_file_url,str(split_path[0])+str("/1.wav"))
                #os.remove(uploaded_file_url)
                #uploaded_file_url = str(split_path[0])+str("/1.wav")


                #load model
                model = tf.keras.models.load_model(SAVED_MODEL_PATH)
                print("hello")
                #model.summary()

                #load the song
                x, sr = librosa.load(song_path, sr = sample_rate)
                song_length = int(librosa.get_duration(filename=song_path))

                prediction_per_part = []

                flag = 0
                if song_length > 30:
                    print("Song is greater than 30 seconds")
                    samples_per_track_30 = sample_rate * song_length
                    parts = int(song_length/30)
                    samples_per_segment_30 = int(samples_per_track_30 / (parts))
                    flag = 1
                    print("Song sliced into "+str(parts)+" parts")
                elif song_length == 30:
                    parts = 1
                    flag = 0
                else:
                    print("Too short, enter a song of length minimum 30 seconds")
                    flag = 2

                for i in range(0,parts):
                    if flag == 1:
                        print("Song snippet ",i+1)
                        start30 = samples_per_segment_30 * i
                        finish30 = start30 + samples_per_segment_30
                        y = x[start30:finish30]
                        #print(len(y))
                    elif flag == 0:
                        y=x
                        print("Song is 30 seconds, no slicing")

                    for n in range(num_segment):
                        start = samples_per_segment * n
                        finish = start + samples_per_segment
                        #print(len(y[start:finish]))
                        mfcc = librosa.feature.mfcc(y[start:finish], sample_rate, n_mfcc = num_mfcc, n_fft = n_fft, hop_length = hop_length)
                        mfcc = mfcc.T
                        #print(mfcc.shape)
                        mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])
                        #print(mfcc.shape)
                        array = model.predict(mfcc)*100
                        array = array.tolist()

                        #find maximum percentage class predicted
                        class_predictions.append(array[0].index(max(array[0])))

                    occurence_dict = {}
                    for i in class_predictions:
                        if i not in occurence_dict:
                            occurence_dict[i] = 1
                        else:
                            occurence_dict[i] +=1

                    max_key = max(occurence_dict, key=occurence_dict.get)
                    prediction_per_part.append(classes[max_key])

                #print(prediction_per_part)
                prediction = max(set(prediction_per_part), key = prediction_per_part.count)
                print(prediction)
                context["prediction"] = prediction

                os.remove(uploaded_file_url)
        else:
            if request.method == 'POST':
                for i in request.POST:
                    if request.POST[i] == "on":
                        input_file_name = str(i)

                if input_file_name == "":
                    context["is_none"] = 1
                    return render(request,"sound_ml/musicgenre.html",context)

                uploaded_file_url = test_data_path + input_file_name + ".wav"
                song_path = str(uploaded_file_url)
                print(uploaded_file_url)

                #load model
                model = tf.keras.models.load_model(SAVED_MODEL_PATH)
                print("hello")
                #model.summary()

                #load the song
                x, sr = librosa.load(song_path, sr = sample_rate)
                song_length = int(librosa.get_duration(filename=song_path))

                prediction_per_part = []

                flag = 0
                if song_length > 30:
                    print("Song is greater than 30 seconds")
                    samples_per_track_30 = sample_rate * song_length
                    parts = int(song_length/30)
                    samples_per_segment_30 = int(samples_per_track_30 / (parts))
                    flag = 1
                    print("Song sliced into "+str(parts)+" parts")
                elif song_length == 30:
                    parts = 1
                    flag = 0
                else:
                    print("Too short, enter a song of length minimum 30 seconds")
                    flag = 2

                for i in range(0,parts):
                    if flag == 1:
                        print("Song snippet ",i+1)
                        start30 = samples_per_segment_30 * i
                        finish30 = start30 + samples_per_segment_30
                        y = x[start30:finish30]
                        #print(len(y))
                    elif flag == 0:
                        y=x
                        print("Song is 30 seconds, no slicing")

                    for n in range(num_segment):
                        start = samples_per_segment * n
                        finish = start + samples_per_segment
                        #print(len(y[start:finish]))
                        mfcc = librosa.feature.mfcc(y[start:finish], sample_rate, n_mfcc = num_mfcc, n_fft = n_fft, hop_length = hop_length)
                        mfcc = mfcc.T
                        #print(mfcc.shape)
                        mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])
                        #print(mfcc.shape)
                        array = model.predict(mfcc)*100
                        array = array.tolist()

                        #find maximum percentage class predicted
                        class_predictions.append(array[0].index(max(array[0])))

                    occurence_dict = {}
                    for i in class_predictions:
                        if i not in occurence_dict:
                            occurence_dict[i] = 1
                        else:
                            occurence_dict[i] +=1

                    max_key = max(occurence_dict, key=occurence_dict.get)
                    prediction_per_part.append(classes[max_key])

                #print(prediction_per_part)
                prediction = max(set(prediction_per_part), key = prediction_per_part.count)
                print(prediction)
                context["prediction"] = prediction


    except:
        if request.method == 'POST' and request.FILES['myfile']:
            os.remove(uploaded_file_url)
            pass

    tf.keras.backend.clear_session()
    return render(request,"sound_ml/musicgenre.html",context)
