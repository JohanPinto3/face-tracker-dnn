import cv2
import numpy as np
import time

CONF_THRESH = 0.5  # show boxes only if confidence >= 50%

MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
CONFIG_PATH = "deploy.prototxt"

def main():
    # Load DNN model
    net = cv2.dnn.readNetFromCaffe(CONFIG_PATH, MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Variables for FPS calculation
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        h, w = frame.shape[:2]

        # Prepare blob and run forward pass
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )
        net.setInput(blob)
        detections = net.forward()

        # Draw detections with confidence %
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < CONF_THRESH:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{conf * 100:.1f}%"
            y_text = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(frame, label, (x1, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # === FPS Calculation ===
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        # Show FPS in top-left corner
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("DNN Face Detection (press q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

