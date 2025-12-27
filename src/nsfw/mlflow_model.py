import pathlib

import mlflow.pyfunc
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms


class NSFWModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path

        # Загружаем конфигурацию для трансформаций (не для модели)
        config_path = pathlib.Path(__file__).parent.parent.parent / "configs"
        config_file = config_path / "mlflow_model" / "mlflow_model_config.yaml"
        if config_file.exists():
            transform_config = OmegaConf.load(config_file)
        else:
            print("Конфиг не найден")

        self.transform_config = transform_config
        self.threshold = transform_config.get("threshold", 0.5)

        # Создаем трансформации из конфига
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        transform_config.image_size.height,
                        transform_config.image_size.width,
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=transform_config.normalize.mean,
                    std=transform_config.normalize.std,
                ),
            ]
        )

    def load_context(self, context):
        from nsfw.model import ConvNextModel

        model_path = context.artifacts.get("model")
        if not model_path:
            print("Модель не загружена")
            return None

        checkpoint = torch.load(model_path, map_location="cpu")
        if "hyper_parameters" in checkpoint:
            config_dict = checkpoint["hyper_parameters"]

        if config_dict is None:
            print("Конфиг не найден")
            return None

        self.model = ConvNextModel.load_from_checkpoint(model_path, config=config_dict)
        self.model.eval()

    def predict(self, context, model_input):
        if self.model is None:
            print("Модель не загружена")
            return None

        image_path = pathlib.Path(model_input)
        if not image_path.exists():
            print(f"Изображение не найдено: {image_path}")
            return None

        image = Image.open(image_path).convert("RGB")

        image_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(image_tensor)
            prob = torch.sigmoid(logits).item()

        prediction = 1 if prob > self.threshold else 0
        class_name = "NSFW" if prediction == 1 else "SFW"

        return {
            "probability": float(prob),
            "prediction": int(prediction),
            "class": class_name,
        }
