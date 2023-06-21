# nemo_asr_inferencer

pip install cython
pip install nemo_toolkit[asr]

Download the checkpoint from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_de_conformer_transducer_large/files# and put it into "checkpoints/conformer_transducer"

Running: python main.py --files <path to directory containing .wav files>