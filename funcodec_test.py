## This scripts test if the funcodec model from the server and ac is the same (though mos listening)


import numpy as np 
import torch 
import torchaudio

code = np.load("fileid_896.npy")
code = torch.from_numpy(code)

print(code.shape)
code = code.cuda()


from funcodec.bin.codec_inference import Speech2Token


codec_model_file = "/public/home/qinxy/bltang/LLM_TTS/egs/exp/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/model.pth"
codec_config_file = "/public/home/qinxy/bltang/LLM_TTS/egs/exp/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/config.yaml"


model = Speech2Token(config_file=codec_config_file, model_file= codec_model_file, device = "cuda")



out = model(code.unsqueeze(0), run_mod="decode")[2].squeeze(0).cpu() # [1,T]

print(f"out shape {out.shape}")

torchaudio.save("out.wav", out, sample_rate = 16000)



