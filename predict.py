# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from PIL import Image
from transparent_background import Remover
import tempfile

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.remover = Remover()

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
    ) -> Path:
        """Run a single prediction on the model"""
        img = Image.open(str(image)).convert('RGB')
        out = self.remover.process(img)
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        Image.fromarray(out).save(out_path)
        return Path(out_path)
