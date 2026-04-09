from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.80

model = YOLO("model/best.pt")

images = [
    "test_images/al.jpg",
    "test_images/h.jpg",
    "test_images/pn.jpg"
]

for img in images:
    print(f"\nTesting image: {img}")

    results = model(img)

    for r in results:
        probs = r.probs
        if probs is not None:
            class_id = int(probs.top1)
            confidence = float(probs.top1conf)
            class_name = r.names[class_id]

            if confidence >= CONFIDENCE_THRESHOLD:
                final_class = class_name
            else:
                final_class = "desconhecido"

            print(f"Predicted: {final_class}")
            print(f"Raw class: {class_name}")
            print(f"Confidence: {confidence:.4f}")