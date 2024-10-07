import cv2
import time
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import requests
import json


db_url = "1"

# 640x480 화면 기준으로 고정된 ID에 해당하는 좌표 배열 (3x3 그리드)
PREDEFINED_LOCATIONS = {
    2: (160, 120),  # 1번 위치 (x, y)
    1: (320, 120),  # 2번 위치 (x, y)
    7: (480, 120),  # 3번 위치 (x, y)
    3: (160, 240),  # 4번 위치 (x, y)
    8: (320, 240),  # 5번 위치 (x, y)
    6: (480, 240),  # 6번 위치 (x, y)
    4: (160, 360),  # 7번 위치 (x, y)
    5: (320, 360),  # 8번 위치 (x, y)
    9: (480, 360),  # 9번 위치 (x, y)
}

def find_closest_id(center_point, predefined_locations, used_ids):
    """탐지된 물체의 중심 좌표를 기준으로 가장 가까운 사전 정의된 좌표의 ID를 반환."""
    min_distance = float('inf')
    closest_id = None
    for obj_id, loc in predefined_locations.items():
        if obj_id in used_ids:
            continue  # 이미 사용된 ID는 건너뜀
        dist = np.linalg.norm(np.array(center_point) - np.array(loc))
        if dist < min_distance:
            min_distance = dist
            closest_id = obj_id
    return closest_id

class ObjectTracker:
    def __init__(self, model_path, db_url=None):
        self.cap = cv2.VideoCapture("test_video.mp4")
        self.model = YOLO(model_path)  # Load YOLOv8 model
        self.track_history = defaultdict(lambda: [])
        self.db_url = db_url
        self.assigned_ids = {}  # 탐지된 물체에 이미 부여된 ID를 저장
        self.used_ids = set()  # 이미 사용된 ID 저장

        # 비디오 저장 설정
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    def track_objects(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            resized = cv2.resize(frame, (640, 480))  # Resize frame for processing
            results = self.model.track(resized, persist=True)

            if len(results) != 0:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                try:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                except:
                    track_ids = []
                annotated_frame = results[0].plot()

                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    center_x = (x1 + x2) // 2  # 물체의 중심 좌표 (x)
                    center_y = (y1 + y2) // 2  # 물체의 중심 좌표 (y)
                    center_point = (center_x, center_y)

                    if track_id not in self.assigned_ids:
                        # 물체에 아직 ID가 부여되지 않았을 경우, 가장 가까운 ID를 찾음
                        assigned_id = find_closest_id(center_point, PREDEFINED_LOCATIONS, self.used_ids)
                        if assigned_id is not None:
                            self.assigned_ids[track_id] = assigned_id
                            self.used_ids.add(assigned_id)
                            print(f"Assigned fixed ID {assigned_id} to track ID {track_id}")
                    else:
                        # 이미 ID가 부여된 경우, 해당 ID 사용
                        assigned_id = self.assigned_ids[track_id]

                    # 경고 메시지를 위한 변위량 계산 (필요시)
                    track = self.track_history[track_id]
                    new_point = (float(center_x), float(center_y))

                    if len(track) >= 10 and new_point != track[-1]:
                        print(f"warning: Object ID {assigned_id} has a new tracking point")

                        # Prepare the data to be sent to the database if needed
                        if self.db_url is not None and assigned_id is not None:  # assigned_id가 None이 아닐 때만 전송
                            data = {"bar_id": int(assigned_id)}  # Convert to native Python int
                            try:
                                json_data = json.dumps(data)
                                response = requests.post(self.db_url, data=json_data, headers={"Content-Type": "application/json"})
                                if response.status_code == 200:
                                    print(f"Data for track ID {assigned_id} successfully sent to the database.")
                                else:
                                    print(f"Failed to send data for track ID {assigned_id}. Status code: {response.status_code}")
                            except Exception as ex:
                                print(ex)

                    track.append(new_point)
                    if len(track) > 10:
                        track.pop(0)

                    # Draw the tracking history on the frame
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(133, 253, 227), thickness=2)

                    # ID 값을 프레임에 표시
                    cv2.putText(annotated_frame, f'ID: {assigned_id}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 2)

            # 비디오 프레임 저장
            self.out.write(annotated_frame)

            # Display the annotated frame
            cv2.imshow('YOLO Object Tracking', annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.out.release()  # 비디오 파일 저장 종료
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Replace the path to your model accordingly
    model_path = "./point-detection/weights/best.pt"
    tracker = ObjectTracker(model_path, db_url=db_url)
    tracker.track_objects()
