"""Webカメラ映像を歪み補正し、クマを検出してFirebaseへ警告を送信する。"""

import threading
import time

import cv2
import firebase_admin
import numpy as np
from firebase_admin import credentials, db
from playsound import playsound
from ultralytics import YOLO

# ================= 設定 =================
MODEL_PATH = "best_final.pt"
CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONFIDENCE_THRESHOLD = 0.60
ALERT_COOLDOWN_SECONDS = 300
ALERT_EXPIRE_SECONDS = 180

# カメラ設置位置の座標
CAMERA_LAT = 35.7611886
CAMERA_LNG = 139.8302372


def initialize_firebase():
    """Firebaseを一度だけ初期化する。"""
    if not firebase_admin._apps:
        cred = credentials.Certificate("firebase_key.json")
        firebase_admin.initialize_app(
            cred,
            {"databaseURL": "https://bear-warning-default-rtdb.firebaseio.com/"},
        )


def find_bear_class_id(model):
    """YOLOモデルのラベルからクラスIDを取得する。"""
    names = model.names
    bear_names = {"bear"}

    for class_id, class_name in names.items():
        if str(class_name).strip().lower() in bear_names:
            return int(class_id)

    raise ValueError(
        "best_final.pt にクマのクラスが見つかりません。"
        f"現在のクラス: {names}。学習時のクラス名を 'bear' に変更するか、"
        "find_bear_class_id() を修正してください。"
    )


def remove_expired_bear_alerts():
    """Firebaseから期限切れのクマ警告を削除する。"""
    ref = db.reference("alerts/bear")
    data = ref.get()
    if not data:
        return

    now = int(time.time())
    for key, value in data.items():
        timestamp = value.get("timestamp")
        if timestamp and now - timestamp > ALERT_EXPIRE_SECONDS:
            ref.child(key).delete()
            print(f"期限切れのクマ警告を削除しました: {key}")


def play_warning_sound():
    """バックグラウンドで警告音を再生する。"""
    playsound("gunshot.wav")
    playsound("dog_bark.wav")


def main():
    initialize_firebase()

    # 学習済みのクマ検出モデルを読み込む。model.namesは上書きしない。
    model = YOLO(MODEL_PATH)
    bear_class_id = find_bear_class_id(model)
    print(f"{MODEL_PATH} を読み込みました。クマのクラス: {bear_class_id} ({model.names[bear_class_id]})")

    # キャリブレーション時に作成したカメラパラメータ。
    camera_matrix = np.loadtxt("K.csv", delimiter=",")
    distortion_coefficients = np.loadtxt("d.csv", delimiter=",")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError(f"インデックス {CAMERA_INDEX} のカメラを開けません。")

    last_alert_time = 0.0
    last_cleanup_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("カメラからフレームを取得できません。")
                break

            # YOLOに入力する前に必ずレンズ歪みを補正する。
            undistorted_frame = cv2.undistort(
                frame, camera_matrix, distortion_coefficients
            )
            results = model(
                undistorted_frame, conf=CONFIDENCE_THRESHOLD, verbose=False
            )

            bear_detected = False
            best_confidence = 0.0
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if class_id == bear_class_id and confidence >= CONFIDENCE_THRESHOLD:
                    bear_detected = True
                    best_confidence = max(best_confidence, confidence)

            current_time = time.time()
            if bear_detected and current_time - last_alert_time >= ALERT_COOLDOWN_SECONDS:
                last_alert_time = current_time
                threading.Thread(target=play_warning_sound, daemon=True).start()

                alert_data = {
                    "animal": "bear",
                    "lat": CAMERA_LAT,
                    "lng": CAMERA_LNG,
                    "confidence": round(best_confidence, 2),
                    "timestamp": int(current_time),
                }
                db.reference("alerts/bear").push(alert_data)
                print("クマの警告をFirebaseへ送信しました。")

            # フレームごとのFirebase呼び出しを避けるため、古い警告は1分ごとに削除する。
            if current_time - last_cleanup_time >= 60:
                remove_expired_bear_alerts()
                last_cleanup_time = current_time

            annotated_frame = results[0].plot()
            cv2.imshow("Bear Warning - qキーで終了", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
