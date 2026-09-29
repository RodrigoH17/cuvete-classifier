import cv2
from collections import Counter
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    VIDEO_PATH,
    CONFIDENCE_THRESHOLD,
    MOTION_THRESHOLD,
    MIN_CONTOUR_AREA,
    EVENT_END_DELAY_FRAMES,
    MIN_EVENT_FRAMES,
    ROI_X,
    ROI_Y,
    ROI_W,
    ROI_H,
    SHOW_VIDEO,
)


def classify_frame(model, frame):
    results = model(frame, verbose=False)

    for r in results:
        probs = r.probs
        if probs is not None:
            class_id = int(probs.top1)
            confidence = float(probs.top1conf)
            class_name = r.names[class_id]

            if confidence >= CONFIDENCE_THRESHOLD:
                return class_name, confidence

            return "desconhecido", confidence

    return "desconhecido", 0.0


def get_final_event_class(predictions):
    if not predictions:
        return "desconhecido"

    count = Counter(predictions)
    return count.most_common(1)[0][0]


def main():
    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("Erro ao abrir vídeo")
        return

    prev_gray = None

    event_active = False
    frames_without_motion = 0
    event_predictions = []
    event_id = 0

    counts = {
        "hamburguer": 0,
        "hamburguer_bovino": 0,
        "peito": 0,
        "pernas_frango": 0,
        "almondegas": 0,
        "almondegas_bovino": 0,
        "carnepicada": 0,
        "carnepicada_bovino": 0,
        "espetadas": 0,
        "desconhecido": 0,
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[ROI_Y:ROI_Y + ROI_H, ROI_X:ROI_X + ROI_W]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        motion_detected = False

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
                    motion_detected = True
                    break

        prev_gray = gray

        # 🔹 INICIAR EVENTO
        if motion_detected and not event_active:
            event_active = True
            frames_without_motion = 0
            event_predictions = []
            event_id += 1
            print(f"\n[EVENT {event_id}] Iniciado")

        # 🔹 DURANTE EVENTO
        if event_active:
            class_name, confidence = classify_frame(model, roi)

            if class_name != "desconhecido":
                event_predictions.append(class_name)

            if motion_detected:
                frames_without_motion = 0
            else:
                frames_without_motion += 1

            # 🔹 FECHAR EVENTO
            if frames_without_motion >= EVENT_END_DELAY_FRAMES:
                event_active = False

                # decidir classe final
                if len(event_predictions) >= MIN_EVENT_FRAMES:
                    final_class = get_final_event_class(event_predictions)
                else:
                    final_class = "desconhecido"

                # 🚫 ignorar eventos vazios
                if len(event_predictions) == 0:
                    print(f"[EVENT {event_id}] Ignorado (sem previsões)")
                else:
                    counts[final_class] += 1
                    print(f"[EVENT {event_id}] Final class: {final_class}")
                    print(f"[EVENT {event_id}] Predictions: {event_predictions}")

                event_predictions = []
                frames_without_motion = 0

        # 🔹 DESENHAR ROI
        color = (0, 255, 0) if event_active else (0, 0, 255)
        cv2.rectangle(frame, (ROI_X, ROI_Y), (ROI_X + ROI_W, ROI_Y + ROI_H), color, 2)

        # 🔹 MOSTRAR CONTADOR
        y = 30
        for cls, value in counts.items():
            cv2.putText(frame, f"{cls}: {value}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 30

        if SHOW_VIDEO:
            cv2.imshow("Covete Classifier - Video Mode", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    # 🔹 RESULTADO FINAL
    print("\n===== CONTAGEM FINAL =====")
    total = 0
    for cls, value in counts.items():
        print(f"{cls}: {value}")
        total += value
    print(f"TOTAL: {total}")


if __name__ == "__main__":
    main()