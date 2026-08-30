import json
import os
from dataclasses import dataclass

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *


@threestudio.register("dummy-prompt-processor")
class DummyPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pretrained_model_name_or_path: str = ""
        prompt: str = ""

    cfg: Config

    def configure(self) -> None:
        # FLUX reference guidance only needs the prompt string; avoid creating
        # legacy Stable Diffusion text-embedding cache entries.
        self.prompt = self.preprocess_prompt(self.cfg.prompt)
        self.negative_prompt = self.cfg.negative_prompt

    def __call__(self):
        return self
