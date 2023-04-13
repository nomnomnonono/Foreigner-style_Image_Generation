import math
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from src.align import ImageAlign
from src.exaggeration_model import LayerSwapGenerator
from torchvision import transforms
from tqdm import tqdm


class perceptual_module(torch.nn.Module):
    def __init__(self):
        import torchvision

        super().__init__()
        perceptual = torchvision.models.vgg16(pretrained=True)
        self.module1_1 = torch.nn.Sequential(*list(perceptual.children())[0][:1])
        self.module1_2 = torch.nn.Sequential(*list(perceptual.children())[0][1:3])
        self.module3_2 = torch.nn.Sequential(*list(perceptual.children())[0][3:13])
        self.module4_2 = torch.nn.Sequential(*list(perceptual.children())[0][13:20])

    def forward(self, x):
        outputs = {}
        out = self.module1_1(x)
        outputs["1_1"] = out
        out = self.module1_2(out)
        outputs["1_2"] = out
        out = self.module3_2(out)
        outputs["3_2"] = out
        out = self.module4_2(out)
        outputs["4_2"] = out

        return outputs


class TO_VGG(object):
    def __init__(self, device="cuda"):
        self.s_mean = (
            torch.from_numpy(np.asarray([0.5, 0.5, 0.5]))
            .view(1, 3, 1, 1)
            .type(torch.FloatTensor)
            .to(device)
        )
        self.s_std = (
            torch.from_numpy(np.asarray([0.5, 0.5, 0.5]))
            .view(1, 3, 1, 1)
            .type(torch.FloatTensor)
            .to(device)
        )
        self.t_mean = (
            torch.from_numpy(np.asarray([0.485, 0.456, 0.406]))
            .view(1, 3, 1, 1)
            .type(torch.FloatTensor)
            .to(device)
        )
        self.t_std = (
            torch.from_numpy(np.asarray([0.229, 0.224, 0.225]))
            .view(1, 3, 1, 1)
            .type(torch.FloatTensor)
            .to(device)
        )

    def __call__(self, t):
        t = (t + 1) / 2
        t = (t - self.t_mean) / self.t_std
        return t


class Invert:
    def __init__(self):
        self.config = OmegaConf.load("config.yaml").invert
        os.makedirs(self.config.result_dir, exist_ok=True)

    def run(self, image):
        g_ema = LayerSwapGenerator(
            self.config.size, self.config.latent, 8, channel_multiplier=2
        ).to(self.config.device)
        checkpoint = torch.load(self.config.checkpoint)
        g_ema.load_state_dict(checkpoint["g_ema"], strict=False)
        g_ema.eval()

        perceptual = perceptual_module().to(self.config.device)
        perceptual.eval()

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

        n = 50000
        samples = 256
        w = []
        for _ in range(n // samples):
            sample_z = self.mixing_noise(
                samples, self.config.latent, 0, device=self.config.device
            )
            w.append(g_ema.style(sample_z))
        w = torch.cat(w, dim=0)
        self.mean_w = w.mean(dim=0)

        align = ImageAlign()
        photo = (
            transform(align(image).convert("RGB")).unsqueeze(0).to(self.config.device)
        )
        self.image_name = image.split("/")[-1].split(".")[0]
        path = self.invert(g_ema, perceptual, photo)
        return path

    def noise_regularize(self, noises):
        loss = 0

        for noise in noises:
            size = noise.shape[2]

            while True:
                loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
                )

                if size <= 8:
                    break

                noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
                noise = noise.mean([3, 5])
                size //= 2

        return loss

    def noise_normalize_(self, noises):
        for noise in noises:
            mean = noise.mean()
            std = noise.std()

            noise.data.add_(-mean).div_(std)

    def make_noise(self, batch, latent_dim, n_noise, device):
        if n_noise == 1:
            return torch.randn(batch, latent_dim, device=device)

        noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

        return noises

    def mixing_noise(self, batch, latent_dim, prob, device):
        return self.make_noise(batch, latent_dim, 1, device)

    def l2loss(self, input1, input2):
        diff = input1 - input2
        diff = diff.pow(2).mean().sqrt().squeeze()
        return diff

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def invert(self, g_ema, perceptual, real_img):
        result = {}
        to_vgg = TO_VGG()
        self.requires_grad(perceptual, True)
        self.requires_grad(g_ema, True)
        log_size = int(math.log(256, 2))
        num_layers = (log_size - 2) * 2 + 1

        w = self.mean_w.clone().detach().to(self.config.device).unsqueeze(0)
        w.requires_grad = True
        wplr = self.config.wlr
        optimizer = torch.optim.Adam(
            [w],
            lr=self.config.wlr,
        )

        print("optimizing w")

        # loop for w
        pbar = range(self.config.w_iterations)
        pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)
        for idx in pbar:
            if idx + 1 % (self.config.w_iterations // 2) == 0:
                for g in optimizer.param_groups:
                    g["lr"] = g["lr"] * self.config.lr_decay_rate
                    wplr = wplr * self.config.lr_decay_rate

            real_img_vgg = to_vgg(real_img)
            t = 1
            w_tilde = w + torch.randn(w.shape, device=self.config.device) * t * t
            fake_img, _ = g_ema([w_tilde], input_is_latent=True, randomize_noise=True)
            fake_img_vgg = to_vgg(fake_img)

            fake_feature = perceptual(fake_img_vgg)
            real_feature = perceptual(real_img_vgg)

            loss_pixel = self.l2loss(fake_img, real_img)

            loss_feature = []
            for (fake_feat, real_feat) in zip(
                fake_feature.values(), real_feature.values()
            ):
                loss_feature.append(self.l2loss(fake_feat, real_feat))
            loss_feature = torch.mean(torch.stack(loss_feature))

            loss = (
                self.config.lambda_l2 * loss_pixel + self.config.lambda_p * loss_feature
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                (
                    f"optimizing w: loss_pixel: {loss_pixel:.4f}; loss_feature: {loss_feature:.4f}"
                )
            )

        result["w"] = w.squeeze().cpu()

        print("optimizing wp")

        # starting point for w : mean w
        wp = w.unsqueeze(1).repeat(1, self.config.num_layers, 1).detach().clone()
        wp.requires_grad = True

        noises = []
        for layer_idx in range(num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2**res, 2**res]
            noises.append(torch.randn(*shape, device=self.config.device).normal_())
            noises[layer_idx].requires_grad = True

        optimizer = torch.optim.Adam(
            [wp] + noises,
            lr=wplr,
        )

        # loop for wp
        pbar = range(self.config.wp_iterations)
        pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)
        for idx in pbar:
            if idx + 1 % (self.config.wp_iterations // 6) == 0:
                for g in optimizer.param_groups:
                    g["lr"] = g["lr"] * self.config.lr_decay_rate
            real_img_vgg = to_vgg(real_img)
            # loss
            t = max(1 - 3 * idx / self.config.wp_iterations, 0)
            wp_tilde = wp + torch.randn(wp.shape, device=self.config.device) * t * t
            fake_img, _ = g_ema(
                wp_tilde, noise=noises, input_is_w_plus=True, randomize_noise=False
            )
            fake_img_vgg = to_vgg(fake_img)

            fake_feature = perceptual(fake_img_vgg)
            real_feature = perceptual(real_img_vgg)

            loss_pixel = self.l2loss(fake_img, real_img)

            loss_feature = []
            for (fake_feat, real_feat) in zip(
                fake_feature.values(), real_feature.values()
            ):
                loss_feature.append(self.l2loss(fake_feat, real_feat))
            loss_feature = torch.mean(torch.stack(loss_feature))

            loss_noise = self.noise_regularize(noises)

            loss = (
                self.config.lambda_l2 * loss_pixel
                + self.config.lambda_p * loss_feature
                + self.config.lambda_noise * loss_noise
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.noise_normalize_(noises)

            # update pbar
            pbar.set_description(
                (
                    f"optimizing wp: loss_pixel: {loss_pixel:.4f}; loss_feature: {loss_feature:.4f}"
                )
            )

        with torch.no_grad():
            fake_img, _ = g_ema(
                wp, noise=noises, input_is_w_plus=True, randomize_noise=False
            )

        result["wp"] = wp.squeeze().cpu()
        result["noise"] = [n.cpu() for n in noises]
        torch.save(result, self.config.result_dir + f"/{self.image_name}.pt")

        return self.config.result_dir + f"/{self.image_name}.pt"


if __name__ == "__main__":
    import glob

    paths = glob.glob("reference/*")
    inv = Invert()
    for path in paths:
        out = inv.run(path)
        print(out)
