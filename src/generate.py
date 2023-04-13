import os

import torch
from omegaconf import OmegaConf
from src.exaggeration_model import StyleCariGAN
from src.invert import Invert
from torchvision import utils
from torchvision.transforms.functional import to_pil_image


class Generator:
    def __init__(self):
        self.config = OmegaConf.load("config.yaml")
        self.invert_config = self.config.invert
        self.config = self.config.generate
        self.reference_file = torch.load(self.config.reference)
        os.makedirs(self.config.result_dir, exist_ok=True)

    def change(self, filepath):
        self.image = filepath

    def run(self, idx):
        ckpt = torch.load(self.config.checkpoint)
        g_ema = StyleCariGAN(
            self.config.size,
            self.config.latent,
            self.config.n_mlp,
            channel_multiplier=self.config.channel_multiplier,
        ).to(self.config.device)
        g_ema.photo_generator.load_state_dict(ckpt["g_ema"], strict=False)
        g_ema.cari_generator.load_state_dict(ckpt["g_ema"], strict=False)
        del ckpt

        if self.config.truncation < 1:
            with torch.no_grad():
                mean_latent = g_ema.photo_generator.mean_latent(
                    self.config.truncation_mean
                )
        else:
            mean_latent = None

        if (
            self.image.endswith(".png")
            or self.image.endswith(".jpg")
            or self.image.endswith
        ):
            inversion_file = os.path.join(
                self.invert_config.result_dir,
                os.path.split(self.image)[-1].split(".")[0] + ".pt",
            )
            if not os.path.exists(inversion_file):
                inv = Invert()
                inversion_file = inv.run(self.image)
            self.image_name = os.path.split(self.image)[-1].split(".")[0]
            img = self.generate(g_ema, mean_latent, inversion_file, int(idx) - 1)
            return img
        else:
            raise ValueError("Unknow Image Format.")

    @torch.no_grad()
    def generate(self, generator, truncation_latent, inversion_file, idx):
        inversion_file = torch.load(inversion_file)
        wp = inversion_file["wp"].to(self.config.device).unsqueeze(0)
        noise = [n.to(self.config.device) for n in inversion_file["noise"]]

        reference = self.reference_file[idx]["wp"].to(self.config.device).unsqueeze(0)
        img = generator(
            wp,
            reference,
            noise=noise,
            input_is_w_plus=True,
            truncation=self.config.truncation,
            truncation_latent=truncation_latent,
        )

        utils.save_image(
            img["result"],
            os.path.join(self.config.result_dir, f"{self.image_name}_{idx}.png"),
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )

        return os.path.join(self.config.result_dir, f"{self.image_name}_{idx}.png")
