import logging
import warnings

import librosa

warnings.filterwarnings('ignore')

# Configure logging at the top of your slicer.py
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Slicer:
    def __init__(self,
                 sr: int,
                 threshold: float = -30.,
                 min_length: int = 3000,
                 min_interval: int = 100,
                 hop_size: int = 20,
                 max_sil_kept: int = 5000):
        if not min_length >= min_interval >= hop_size:
            raise ValueError('The following condition must be satisfied: min_length >= min_interval >= hop_size')
        if not max_sil_kept >= hop_size:
            raise ValueError('The following condition must be satisfied: max_sil_kept >= hop_size')
        min_interval = sr * min_interval / 1000
        self.sr = sr
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)


    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)]


    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = librosa.to_mono(waveform)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            # Return the entire audio as a single chunk
            return [(0, waveform)]
            
        rms_list = librosa.feature.rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start: i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        
        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        
        # Apply and return slices.
        if len(sil_tags) == 0:
            # Return the entire audio as a single chunk if no silence detected
            return [(0, waveform)]
        
        # Extract non-silence chunks
        non_silence_chunks = []
        
        # Add first non-silence chunk if it exists
        if sil_tags[0][0] > 0:
            start_pos = 0
            end_frame = sil_tags[0][0]
            chunk = self._apply_slice(waveform, 0, end_frame)
            non_silence_chunks.append((start_pos, chunk))
        
        # Add middle non-silence chunks
        for i in range(1, len(sil_tags)):
            start_frame = sil_tags[i-1][1]
            end_frame = sil_tags[i][0]
            if start_frame < end_frame:  # Only add if there's actual non-silence content
                start_pos = start_frame * self.hop_size
                chunk = self._apply_slice(waveform, start_frame, end_frame)
                non_silence_chunks.append((start_pos, chunk))
        
        # Add last non-silence chunk if it exists
        if sil_tags[-1][1] * self.hop_size < len(waveform):
            start_frame = sil_tags[-1][1]
            start_pos = start_frame * self.hop_size
            chunk = self._apply_slice(waveform, start_frame, total_frames)
            non_silence_chunks.append((start_pos, chunk))
        
        for i, (start_pos, chunk) in enumerate(non_silence_chunks):
            # Calculate start and end times in seconds
            start_time_sec = start_pos / self.sr
            end_time_sec = start_pos / self.sr + len(chunk) / self.sr if len(chunk.shape) == 1 else start_pos / self.sr + chunk.shape[1] / self.sr
            duration_sec = end_time_sec - start_time_sec
            
            # Format start and end times as mm:ss
            start_min, start_sec = divmod(start_time_sec, 60)
            end_min, end_sec = divmod(end_time_sec, 60)
            
            # Log the information
            logger.info(f"Chunk {i}: Start={int(start_min):02d}:{start_sec:05.2f}, End={int(end_min):02d}:{end_sec:05.2f}, Duration={duration_sec:.2f}s")
        
        return non_silence_chunks



def main():
    import os.path
    from argparse import ArgumentParser
    import librosa
    import soundfile
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument('audio', type=str, help='The audio file or directory to be sliced')
    parser.add_argument('--out', type=str, help='Output directory of the sliced audio clips')
    parser.add_argument('--db_thresh', type=float, required=False, default=-30,
                        help='The dB threshold for silence detection')
    parser.add_argument('--min_length', type=int, required=False, default=3000,
                        help='The minimum milliseconds required for each sliced audio clip')
    parser.add_argument('--min_interval', type=int, required=False, default=100,
                        help='The minimum milliseconds for a silence part to be sliced')
    parser.add_argument('--hop_size', type=int, required=False, default=20,
                        help='Frame length in milliseconds')
    parser.add_argument('--max_sil_kept', type=int, required=False, default=5000,
                        help='The maximum silence length kept around the sliced clip, presented in milliseconds')
    args = parser.parse_args()
    
    # Determine if the input is a file or directory
    audio_path = Path(args.audio)
    is_directory = audio_path.is_dir()
    
    # Prepare output directory
    out = args.out
    if out is None:
        if is_directory:
            out = os.path.abspath(args.audio)
        else:
            out = os.path.dirname(os.path.abspath(args.audio))
    
    if not os.path.exists(out):
        os.makedirs(out)
    
    # Audio file extensions to process
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    # Process a single file or all files in a directory
    if is_directory:
        logger.info(f"Processing all audio files in directory: {args.audio}")
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(audio_path.glob(f'*{ext}')))
        
        if not audio_files:
            logger.warning(f"No audio files found in {args.audio}")
            return
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        for audio_file in audio_files:
            process_audio_file(audio_file, out, args)
    else:
        # Process a single audio file
        logger.info(f"Processing single audio file: {args.audio}")
        process_audio_file(audio_path, out, args)

def process_audio_file(audio_file, out_dir, args):
    """Process a single audio file with the given parameters"""
    import os.path
    import librosa
    import soundfile
    
    try:
        logger.info(f"Loading audio file: {audio_file}")
        audio, sr = librosa.load(str(audio_file), sr=None, mono=False)
        
        slicer = Slicer(
            sr=sr,
            threshold=args.db_thresh,
            min_length=args.min_length,
            min_interval=args.min_interval,
            hop_size=args.hop_size,
            max_sil_kept=args.max_sil_kept
        )
        
        # Get non-silence chunks with their positions
        chunks_with_pos = slicer.slice(audio)
        
        file_basename = os.path.basename(str(audio_file)).rsplit('.', maxsplit=1)[0]
        logger.info(f"Saving {len(chunks_with_pos)} non-silence audio chunks from {file_basename}...")
        
        for i, (pos, chunk) in enumerate(chunks_with_pos):
            if len(chunk.shape) > 1:
                chunk = chunk.T
            
            output_file = os.path.join(out_dir, f'{file_basename}_{i}_pos_{pos}.wav')
            soundfile.write(output_file, chunk, sr)
        
        logger.info(f"Finished processing {audio_file}")
    except Exception as e:
        logger.error(f"Error processing {audio_file}: {str(e)}")


if __name__ == '__main__':
    main()
