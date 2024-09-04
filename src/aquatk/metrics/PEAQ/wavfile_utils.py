import numpy as np
import struct
import soundfile as sf
from scipy.io import wavfile
import math
import wave


def read_wav_blocks(filename, block_size=2048, overlap=1024):
    blocks = []
    with wave.open(filename, "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        n_blocks = 1 + (n_frames - block_size) // (block_size - overlap)
        # Determine the data type based on the sample width
        if sample_width == 1:
            dtype = "u1"  # 8-bit PCM
        elif sample_width == 2:
            dtype = "i2"  # 16-bit PCM
        elif sample_width == 3:
            dtype = "i3"  # 24-bit PCM
        elif sample_width == 4:
            dtype = "i4"  # 32-bit PCM
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Read blocks
        for _ in range(n_blocks):
            # Read block_size frames
            frames = wav_file.readframes(block_size)
            if len(frames) < block_size * sample_width:
                # If we've reached the end of the file, we can break out of the loop.
                break

            # Convert byte data to numpy array

            block = np.frombuffer(frames, dtype=dtype)

            block = block.reshape(-1, n_channels)

            blocks.append(block)

            # Rewind for overlap
            wav_file.setpos(wav_file.tell() - overlap)
    print(np.array(blocks).dtype)
    print(np.array(blocks).max(), np.array(blocks).min())
    return blocks
