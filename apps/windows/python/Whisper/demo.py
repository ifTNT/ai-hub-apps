# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import numpy as np
from datetime import datetime

from model import WhisperBaseEnONNX
from qai_hub_models.models.whisper_base_en import App as WhisperApp
from qai_hub_models.models._shared.whisper.demo import load_demo_audio


def load_audio_file(filepath: str) -> tuple[np.array, int]:
    import audio2numpy as a2n  # import here, as this requires ffmpeg to be installed on host machine

    audio, audio_sample_rate = a2n.audio_from_file(filepath)
    audio = np.mean(audio, axis=1)

    return audio, audio_sample_rate


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler="error",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Path to audio file that needs to be tested. Only .mp3 are supported.",
    )
    args = parser.parse_args()
    # Input files
    encoder_path = "build/whisper_base_en/WhisperEncoderInf.onnx"
    decoder_path = "build/whisper_base_en/WhisperDecoderInf.onnx"

    # Load whisper model
    print("Loading model...")
    whisper = WhisperApp(WhisperBaseEnONNX(encoder_path, decoder_path))

    # Execute Whisper Model
    print("Before transcription: " + str(datetime.now().astimezone()))
    if args.audio_path is not None:
        audio, audio_sample_rate = load_audio_file(args.audio_path)
    else:
        audio, audio_sample_rate = load_demo_audio()
    text = whisper.transcribe(audio, audio_sample_rate)
    print("After transcription: " + str(datetime.now().astimezone()))
    with open("transcript.txt", "w") as file:
        file.write(text)
    print("After writing file: " + str(datetime.now().astimezone()))


if __name__ == "__main__":
    main()
