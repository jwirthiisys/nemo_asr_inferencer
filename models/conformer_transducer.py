from .asr_net import AsrNet
import nemo.collections.asr as nemo_asr
from typing import Dict

class ConformerTransducer(AsrNet):
    def __init__(self, device) -> None:
        super().__init__(device)
        self.model = nemo_asr.models.EncDecRNNTBPEModel.restore_from("checkpoints/conformer_transducer/stt_de_conformer_transducer_large.nemo").to(self.device)
    
    def batch_inference(self, inputs):
        outputs = self.model.transcribe(paths2audio_files=inputs, batch_size=1, return_hypotheses=False)
        return outputs[0]
    
    def inference(self, input) -> str:
        transcriptions = self.model.transcribe(paths2audio_files=[input], batch_size=1, return_hypotheses=False)
        return transcriptions[0][0]
    
    def train(self):
        raise NotImplementedError()