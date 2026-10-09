import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from utils import device, load_image, save_image
from losses import (load_vgg, get_features, gram_matrix,
                    content_loss, style_loss, STYLE_LAYERS)
from model import TransformerNet

DATA_DIR = "data/val2017"
STYLE_IMAGE = "images/bridge.jpg"
TEST_IMAGE = "images/pelicans.jpeg"
IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 2
LR = 1e-3
CONTENT_WEIGHT = 1e5
STYLE_WEIGHT = 1e10
LOG_EVERY = 50
SAMPLE_EVERY = 500
RUN_NAME = "bridge_sw1e10"


class ImageDataset(Dataset):
    def __init__(self, folder, size):
        self.paths = sorted(Path(folder).glob("*.jpg"))
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)


def main():
    Path("checkpoints").mkdir(exist_ok=True)
    Path(f"samples/{RUN_NAME}").mkdir(parents=True, exist_ok=True)

    dataset = ImageDataset(DATA_DIR, IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, drop_last=True)
    print(f"{len(dataset)} görüntü, epoch başına {len(loader)} adım")

    vgg = load_vgg(device)
    model = TransformerNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    style = load_image(STYLE_IMAGE)
    with torch.no_grad():
        style_features = get_features(style, vgg)
        style_grams = {name: gram_matrix(style_features[name]) for name in STYLE_LAYERS}

    test_image = load_image(TEST_IMAGE)

    step = 0
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for batch in loader:
            batch = batch.to(device)

            output = model(batch)

            with torch.no_grad():
                content_features = get_features(batch, vgg)
            output_features = get_features(output, vgg)

            c_loss = CONTENT_WEIGHT * content_loss(output_features, content_features)
            s_loss = STYLE_WEIGHT * style_loss(output_features, style_grams)
            loss = c_loss + s_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % LOG_EVERY == 0:
                elapsed = time.time() - start
                print(f"epoch {epoch+1} | adım {step} | içerik {c_loss.item():.1f} | "
                      f"stil {s_loss.item():.1f} | toplam {loss.item():.1f} | {elapsed:.0f} sn")

            if step % SAMPLE_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    save_image(model(test_image), f"samples/{RUN_NAME}/step_{step:05d}.jpg")
                model.train()

        torch.save(model.state_dict(), f"checkpoints/{RUN_NAME}_epoch_{epoch+1}.pth")

    print("Eğitim bitti.")


if __name__ == "__main__":
    main()