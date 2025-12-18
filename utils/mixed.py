from pydub import AudioSegment

path1 = "path/to/first/audio.wav"
path2 = "path/to/second/audio.wav"
output_path = "path/to/output/mixed_audio.wav"
track1 = AudioSegment.from_file(path1)
track2 = AudioSegment.from_file(path2)

# 假设同步开始，则简单相加
mixed = track1.overlay(track2)

# 导出
mixed.export(output_path, format="wav")
