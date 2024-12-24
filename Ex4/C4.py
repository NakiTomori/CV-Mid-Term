# Import necessary libraries
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from detr.models.detr import build_detr
from yolov4.models import YOLOv4
from yolov4.utils.datasets import YOLODataset
from yolov4.utils.train import train_yolov4
from faster_rcnn.models import FasterRCNN
from faster_rcnn.utils.train import train_faster_rcnn

# Define dataset paths and parameters
data_dir = "./coco_dataset"
batch_size = 16
num_classes = 80  # MS COCO class count
image_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load COCO dataset
dataset = datasets.CocoDetection(
    root=f"{data_dir}/train2017",
    annFile=f"{data_dir}/annotations/instances_train2017.json",
    transform=transform,
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# DETR training setup
def train_detr():
    model = build_detr(num_classes=num_classes)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(50):  # Adjust epoch count as needed
        model.train()
        for images, targets in dataloader:
            images = images.to(device)
            targets = [{"labels": t["category_id"].to(device)} for t in targets]

            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/50], Loss: {loss.item()}")

# Faster R-CNN training setup
def train_faster_rcnn():
    model = FasterRCNN(num_classes=num_classes)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(20):
        model.train()
        for images, targets in dataloader:
            images = images.to(device)
            targets = [{"labels": t["category_id"].to(device)} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/20], Loss: {losses.item()}")

# YOLOv4 training setup
def train_yolov4():
    model = YOLOv4(num_classes=num_classes, image_size=image_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    yolo_dataset = YOLODataset(data_dir)
    yolo_dataloader = DataLoader(yolo_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    train_yolov4(model, yolo_dataloader, optimizer, epochs=50)

# Run training functions
if __name__ == "__main__":
    print("Training DETR...")
    train_detr()

    print("Training Faster R-CNN...")
    train_faster_rcnn()

    print("Training YOLOv4...")
    train_yolov4()
