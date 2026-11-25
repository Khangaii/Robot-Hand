# visual studio 에서 직접 실행하는 code
# python "camera_hand_mid50_v1.py" --port COM11 -> 아두이노 연결 후 port 값 수정하면 카메라 열리면서 mediapipe 실행됨.
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 코드와 같은 폴더에 있는 모델 파일
model_path = Path(__file__).resolve().with_name("hand_landmarker.task")

# 손 관절을 연결하는 선
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17),
    (17, 18), (18, 19), (19, 20),
]

# 모델을 메모리로 읽어서 사용
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_buffer=model_path.read_bytes()
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)

# 노트북 카메라 열기
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap.release()
    raise RuntimeError("카메라를 열 수 없습니다.")

last_timestamp = -1

try:
    with vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            success, frame = cap.read()

            if not success:
                print("카메라 영상을 읽지 못했습니다.")
                break

            # 거울처럼 좌우 반전
            frame = cv2.flip(frame, 1)

            # MediaPipe에 맞는 색상 순서로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # 각 영상 프레임의 시간값
            timestamp = max(
                time.monotonic_ns() // 1_000_000,
                last_timestamp + 1,
            )
            last_timestamp = timestamp

            # 손 관절 찾기
            result = detector.detect_for_video(image, timestamp)
            height, width = frame.shape[:2]

            for hand in result.hand_landmarks:
                points = [
                    (int(point.x * width), int(point.y * height))
                    for point in hand
                ]

                # 관절을 초록색 선으로 연결
                for start, end in connections:
                    cv2.line(
                        frame, points[start], points[end],
                        (0, 255, 0), 2
                    )

                # 관절마다 빨간 점과 번호 표시
                for number, (x, y) in enumerate(points):
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(
                        frame, str(number), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1
                    )

            cv2.imshow("Hand Tracking - ESC to quit", frame)

            # Esc 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
