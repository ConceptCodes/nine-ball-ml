from ultralytics import YOLO

print("Loading the pretrained model...")
model = YOLO("models/yolo11s.pt")

print("Starting the training process...")
print("Dataset info: 582 training images, 54 validation images")
print("Image resolution: 640x360")

results = model.train(
    data="src/config/training.yaml",
    imgsz=640,
    epochs=150,  # Increase epochs for better convergence
    batch=16,  # Increase batch size if GPU memory allows
    name="pool-ball-detection-v2",
    plots=True,
    amp=True,
    patience=25,  # Increase patience for longer training
    save_period=10,
    val=True,
    lr0=0.005,  # Reduce initial learning rate for finer adjustments
    lrf=0.01,  # Lower final learning rate
    momentum=0.937,
    weight_decay=0.001,  # Increase weight decay for better regularization
    warmup_epochs=5,  # Increase warmup
    warmup_momentum=0.8,
    box=10.0,  # Increase box loss gain for better localization
    cls=0.5,
    dfl=2.0,  # Increase DFL loss for better bounding box regression
    pose=12.0,
    kobj=1.0,
    label_smoothing=0.1,  # Add label smoothing to reduce overconfidence
    nbs=64,
    # Reduce augmentation for more precise learning
    hsv_h=0.01,  # Reduce HSV augmentation
    hsv_s=0.5,   # Reduce saturation changes
    hsv_v=0.3,   # Reduce value changes
    degrees=5.0,  # Add slight rotation for robustness
    translate=0.05,  # Reduce translation
    scale=0.2,   # Reduce scaling augmentation
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.3,  # Reduce horizontal flip probability
    mosaic=0.8,  # Reduce mosaic augmentation
    mixup=0.1,   # Add small amount of mixup
    copy_paste=0.1,  # Add copy-paste for data diversity
)

print("Training process completed.")
print(f"Best model saved at: runs/detect/pool-ball-detection/weights/best.pt")
print(f"Last model saved at: runs/detect/pool-ball-detection/weights/last.pt")
