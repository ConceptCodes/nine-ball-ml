from ultralytics import YOLO

print("Loading the pretrained model...")
model = YOLO("models/yolo11s.pt")

print("Starting the training process...")
print("Dataset info: 582 training images, 54 validation images")
print("Image resolution: 640x360")

results = model.train(
    data="src/config/training.yaml",
    imgsz=640,  # Matches image width (640x360)
    epochs=100,  # Increased for better learning with your dataset size
    batch=8,  # Reduced batch size for better stability with smaller dataset
    name="pool-ball-detection",
    plots=True,
    amp=True,  # Automatic Mixed Precision for faster training
    patience=15,  # Early stopping if no improvement for 15 epochs
    save_period=10,  # Save checkpoint every 10 epochs
    val=True,  # Enable validation during training
    lr0=0.01,  # Initial learning rate
    lrf=0.1,  # Final learning rate (lr0 * lrf)
    momentum=0.937,  # SGD momentum
    weight_decay=0.0005,  # Weight decay for regularization
    warmup_epochs=3,  # Warmup epochs
    warmup_momentum=0.8,  # Warmup momentum
    box=7.5,  # Box loss gain
    cls=0.5,  # Class loss gain
    dfl=1.5,  # DFL loss gain
    pose=12.0,  # Pose loss gain (not used for detection)
    kobj=1.0,  # Keypoint obj loss gain (not used for detection)
    label_smoothing=0.0,  # Label smoothing epsilon
    nbs=64,  # Nominal batch size for scaling
    hsv_h=0.015,  # HSV-Hue augmentation (fraction)
    hsv_s=0.7,  # HSV-Saturation augmentation (fraction)
    hsv_v=0.4,  # HSV-Value augmentation (fraction)
    degrees=0.0,  # Rotation augmentation (degrees)
    translate=0.1,  # Translation augmentation (fraction)
    scale=0.5,  # Scaling augmentation gain
    shear=0.0,  # Shear augmentation (degrees)
    perspective=0.0,  # Perspective augmentation (probability)
    flipud=0.0,  # Vertical flip augmentation (probability)
    fliplr=0.5,  # Horizontal flip augmentation (probability)
    mosaic=1.0,  # Mosaic augmentation (probability)
    mixup=0.0,  # Mixup augmentation (probability)
    copy_paste=0.0,  # Copy-paste augmentation (probability)
)

print("Training process completed.")
print(f"Best model saved at: runs/detect/pool-ball-detection/weights/best.pt")
print(f"Last model saved at: runs/detect/pool-ball-detection/weights/last.pt")
