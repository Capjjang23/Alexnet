import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pydub import AudioSegment

i = 0;

for dirname, _, filenames in os.walk("dataset/m4a"):
    for filename in filenames:
        m4a_file = os.path.join(dirname, filename)
        start_pos = m4a_file.rfind('/')
        alpha = m4a_file[start_pos-1:start_pos]

        if (i != 0):
            if (before_alpha != alpha):
                i = 1
            else:
                i += 1
        else:
            i += 1

        before_alpha = alpha

        # print(start_pos, end_pos, imsi_breed)

        wav_filename = "dataset/wav/" + str(alpha) + "/" + str(alpha) + str(i) + ".wav"
        print(wav_filename)
        track = AudioSegment.from_file(m4a_file, format='m4a')
        file_handle = track.export(wav_filename, format='wav')