from pydub import AudioSegment
from pydub.utils import make_chunks
import os

input_path = "pure_sound.wav"
output_folder = "dataset"


audio = AudioSegment.from_file(input_path, "wav")

size = 15000  #切割的毫秒数

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
 
chunks = make_chunks(audio, size) 
for i, chunk in enumerate(chunks):
    chunk_name = os.path.join(output_folder, "new-{0}.wav".format(i))
    print(chunk_name)
    chunk.export(chunk_name, format="wav")
