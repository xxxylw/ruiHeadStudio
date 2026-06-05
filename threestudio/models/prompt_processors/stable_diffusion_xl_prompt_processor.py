import os
from dataclasses import dataclass
from typing import Callable

import torch
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

import threestudio
from threestudio.models.prompt_processors.base import (
    DirectionConfig,
    PromptProcessor,
    hash_prompt,
    shift_azimuth_deg,
)
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *


@dataclass
class SDXLDirectionConfig:
    name: str
    condition: Callable[
        [Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]],
        Float[Tensor, "B"],
    ]


@dataclass
class SDXLPromptProcessorOutput:
    prompt_embeds: Float[Tensor, "1 N 2048"]
    negative_prompt_embeds: Float[Tensor, "1 N 2048"]
    pooled_prompt_embeds: Float[Tensor, "1 1280"]
    negative_pooled_prompt_embeds: Float[Tensor, "1 1280"]
    prompt_embeds_vd: Float[Tensor, "Nv N 2048"]
    negative_prompt_embeds_vd: Float[Tensor, "Nv N 2048"]
    pooled_prompt_embeds_vd: Float[Tensor, "Nv 1280"]
    negative_pooled_prompt_embeds_vd: Float[Tensor, "Nv 1280"]
    directions: list[SDXLDirectionConfig]
    direction2idx: dict[str, int]

    def get_sdxl_text_embeddings(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        view_dependent_prompting: bool = True,
    ) -> dict[str, Float[Tensor, "..."]]:
        batch_size = elevation.shape[0]

        if view_dependent_prompting:
            direction_idx = torch.zeros_like(elevation, dtype=torch.long)
            for direction in self.directions:
                direction_idx[
                    direction.condition(elevation, azimuth, camera_distances)
                ] = self.direction2idx[direction.name]

            prompt_embeds = self.prompt_embeds_vd[direction_idx]
            negative_prompt_embeds = self.negative_prompt_embeds_vd[direction_idx]
            pooled_prompt_embeds = self.pooled_prompt_embeds_vd[direction_idx]
            negative_pooled_prompt_embeds = self.negative_pooled_prompt_embeds_vd[
                direction_idx
            ]
        else:
            prompt_embeds = self.prompt_embeds.expand(batch_size, -1, -1)
            negative_prompt_embeds = self.negative_prompt_embeds.expand(
                batch_size, -1, -1
            )
            pooled_prompt_embeds = self.pooled_prompt_embeds.expand(batch_size, -1)
            negative_pooled_prompt_embeds = self.negative_pooled_prompt_embeds.expand(
                batch_size, -1
            )

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        }


@threestudio.register("stable-diffusion-xl-prompt-processor")
class StableDiffusionXLPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pass

    cfg: Config

    def configure_text_encoder(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.tokenizer_2 = AutoTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="tokenizer_2"
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device)
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder_2"
        ).to(self.device)

        for module in [self.text_encoder, self.text_encoder_2]:
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.tokenizer_2
        del self.text_encoder
        del self.text_encoder_2
        cleanup()

    @staticmethod
    def _encode_prompt_with_models(
        prompt: list[str],
        tokenizer,
        tokenizer_2,
        text_encoder,
        text_encoder_2,
    ) -> dict[str, Tensor]:
        tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens_2 = tokenizer_2(
            prompt,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            enc_1 = text_encoder(
                tokens.input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )
            enc_2 = text_encoder_2(
                tokens_2.input_ids.to(text_encoder_2.device),
                output_hidden_states=True,
            )

        prompt_embeds = torch.cat(
            [
                enc_1.hidden_states[-2].to(enc_2.hidden_states[-2].device),
                enc_2.hidden_states[-2],
            ],
            dim=-1,
        )
        pooled_prompt_embeds = enc_2.text_embeds
        return {
            "prompt_embeds": prompt_embeds.detach().cpu(),
            "pooled_prompt_embeds": pooled_prompt_embeds.detach().cpu(),
        }

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        tokenizer_2 = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer_2"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            device_map="auto",
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            device_map="auto",
        )

        encoded = StableDiffusionXLPromptProcessor._encode_prompt_with_models(
            prompts, tokenizer, tokenizer_2, text_encoder, text_encoder_2
        )
        for index, prompt in enumerate(prompts):
            torch.save(
                {
                    "prompt_embeds": encoded["prompt_embeds"][index],
                    "pooled_prompt_embeds": encoded["pooled_prompt_embeds"][index],
                },
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

        del text_encoder
        del text_encoder_2
        cleanup()

    def load_sdxl_from_cache(self, prompt):
        cache_path = os.path.join(
            self._cache_dir,
            f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"SDXL text embedding file {cache_path} for model "
                f"{self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] not found."
            )
        cached = torch.load(cache_path, map_location=self.device)
        if not isinstance(cached, dict):
            raise TypeError(f"Expected SDXL prompt cache dict at {cache_path}")
        return cached

    def load_text_embeddings(self):
        from threestudio.utils.misc import barrier

        barrier()

        prompt = self.load_sdxl_from_cache(self.prompt)
        negative = self.load_sdxl_from_cache(self.negative_prompt)
        prompt_vd = [self.load_sdxl_from_cache(prompt) for prompt in self.prompts_vd]
        negative_vd = [
            self.load_sdxl_from_cache(prompt) for prompt in self.negative_prompts_vd
        ]

        self.prompt_embeds = prompt["prompt_embeds"][None, ...]
        self.negative_prompt_embeds = negative["prompt_embeds"][None, ...]
        self.pooled_prompt_embeds = prompt["pooled_prompt_embeds"][None, ...]
        self.negative_pooled_prompt_embeds = negative["pooled_prompt_embeds"][
            None, ...
        ]
        self.prompt_embeds_vd = torch.stack(
            [item["prompt_embeds"] for item in prompt_vd], dim=0
        )
        self.negative_prompt_embeds_vd = torch.stack(
            [item["prompt_embeds"] for item in negative_vd], dim=0
        )
        self.pooled_prompt_embeds_vd = torch.stack(
            [item["pooled_prompt_embeds"] for item in prompt_vd], dim=0
        )
        self.negative_pooled_prompt_embeds_vd = torch.stack(
            [item["pooled_prompt_embeds"] for item in negative_vd], dim=0
        )

    def __call__(self) -> SDXLPromptProcessorOutput:
        return SDXLPromptProcessorOutput(
            prompt_embeds=self.prompt_embeds,
            negative_prompt_embeds=self.negative_prompt_embeds,
            pooled_prompt_embeds=self.pooled_prompt_embeds,
            negative_pooled_prompt_embeds=self.negative_pooled_prompt_embeds,
            prompt_embeds_vd=self.prompt_embeds_vd,
            negative_prompt_embeds_vd=self.negative_prompt_embeds_vd,
            pooled_prompt_embeds_vd=self.pooled_prompt_embeds_vd,
            negative_pooled_prompt_embeds_vd=self.negative_pooled_prompt_embeds_vd,
            directions=[
                SDXLDirectionConfig(
                    direction.name,
                    direction.condition,
                )
                for direction in self.directions
            ],
            direction2idx=self.direction2idx,
        )
