from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

import settings

instance: YOLO = YOLO(settings.BASE_MODEL)

if not Path(settings.MODEL_PATH).is_file():
    instance.train(data='flowers', epochs=settings.EPOCHS)

model: YOLO = YOLO(settings.MODEL_PATH)
folders: iter = Path('datasets/flowers/train').iterdir()
folder_names: list[str] = [folder.name for folder in folders]

result: list = model('sunflower.png')

print(folder_names[result[0].probs.top1], result[0].probs.top5)
