# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import re
from PIL import Image
from threading import Thread
from transformers import TextIteratorStreamer
from huggingface_hub import snapshot_download
from moondream import VisionEncoder, TextModel

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        model_path = snapshot_download("vikhyatk/moondream1", cache_dir="checkpoints")
        self.vision_encoder = VisionEncoder(model_path)
        self.text_model = TextModel(model_path)
        
    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
        prompt: str = Input(description="Prompt to use for generation", default=None),
    ) -> str:
        """Run a single prediction on the model"""
        image = Image.open(image).convert("RGB")
        image_embeds = self.vision_encoder(image)

        if prompt is None:
            question = input("> ")
            streamer = TextIteratorStreamer(self.text_model.tokenizer, skip_special_tokens=True)
            generation_kwargs = dict(
                image_embeds=image_embeds, question=question, streamer=streamer
            )
            thread = Thread(target=self.text_model.answer_question, kwargs=generation_kwargs)
            thread.start()
            buffer = ""
            for new_text in streamer:
                buffer += new_text
                if not new_text.endswith("<") and not new_text.endswith("END"):
                    print(buffer, end="", flush=True)
                    buffer = ""

            result = re.sub("<$", "", re.sub("END$", "", buffer))
            return result
        else:
            question = prompt
            result = self.text_model.answer_question(image_embeds, question)
            return result
