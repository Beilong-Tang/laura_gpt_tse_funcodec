from torch import nn
import torch
from argparse import Namespace
from funcodec.torch_utils.load_pretrained_model import load_pretrained_model
from funcodec.tasks.text2audio_generation import Text2AudioGenTask
from funcodec.utils.misc import statistic_model_parameters
from funcodec.bin.codec_inference import Speech2Token


class TSExtraction:
    def __init__(self, args: Namespace, model_ckpt: str, device, only_lm, logger):
        # Load Laura GPT Model #
        model: nn.Module = Text2AudioGenTask.build_model(args)
        model.to(device)
        for p in args.init_param:
            load_pretrained_model(
                model=model,
                init_param=p,
                ignore_init_mismatch=True,
                # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
                #   in PyTorch<=1.4
                map_location=device,
            )
        logger.info("model: {}".format(model))
        logger.info(
            "model parameter number: {}".format(statistic_model_parameters(model))
        )

        # Load Ckpt #
        ckpt = torch.load(model_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self.model = model
        logger.info("model loaded successfully!")

        # Load Codec Model
        codec_kwargs = dict(
            config_file=args["codec_config_file"],
            model_file=args["codec_model_file"],
            device=device,
        )
        self.codec_model = Speech2Token.from_pretrained(
            model_tag=None,
            **codec_kwargs,
        )

        # sampling and beam_size
        self.sampling = args.sampling
        self.beam_size = args.beam_size
        self.only_lm = only_lm

    @torch.no_grad()
    def __call__(self, mix_mel:torch.Tensor, ref_mel:torch.Tensor, ref_codec:torch.Tensor):
        """
        This function can also be used as TSE Inference.
        mix_mel the mep spec of the mixture: [1, T, D]
        ref_mel is the reference mel : [1, T, D]
        ref_codec is the [T, N_Q], the n_q can be actually larger than the predict_nq
        """
        text = torch.cat([ref_mel, mix_mel], dim = 1) # [1,T',D]
        ref_codec = ref_codec[:, :self.model.predict_nq] # [T, 2]
        ref_codec = ref_codec.tolist()
        continual = ref_codec
        continual_length = len(continual)

        # 1. Encode Text(Mel)
        text_lens = torch.tensor([text.size(1)], dtype=torch.long, device=text.device)
        text_outs, text_out_lens = self.model.encode(text, text_lens)
        # 2. decode first codec group
        decoded_codec = self.model.decode_codec(
            text_outs,
            text_out_lens,
            max_length=30 * 25,
            sampling=self.sampling,
            beam_size=self.beam_size,
            continual=continual,
        )
        if self.only_lm:
            _, _, gen_speech_only_lm, _ = self.codec_model(
                decoded_codec[:, continual_length:], bit_width=None, run_mod="decode"
            )
            ret_val = dict(gen=gen_speech_only_lm)
            return (ret_val, decoded_codec)
        else:
            # 3. predict embeddings
            gen_speech = self.model.syn_audio(
                decoded_codec,
                text_outs,
                text_out_lens,
                self.codec_model,
                continual_length=continual_length,
            )
            ret_val = dict(
                gen=gen_speech,
                # gen_only_lm=gen_speech_only_lm,
            )

            return (
                ret_val,
                decoded_codec,
            )  # {'gen':[1,1,T] }, [1,T,n_q]

