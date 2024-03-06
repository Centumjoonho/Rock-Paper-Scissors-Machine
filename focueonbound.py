import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# OpenCV 캡처 객체 생성
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    success, img = cap.read()
    
    # MediaPipe 처리
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 피사체 리스트 초기화
    subjects = []
    
    # 각 랜드마크에 대한 처리
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
            subjects.append((x, y))
        
        # 바운딩 박스 계산
        x_min = min(subjects, key=lambda x: x[0])[0]
        x_max = max(subjects, key=lambda x: x[0])[0]
        y_min = min(subjects, key=lambda x: x[1])[1]
        y_max = max(subjects, key=lambda x: x[1])[1]
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        bbox_area = bbox_width * bbox_height
        
        # 가장 큰 피사체 찾기
        max_subject_bbox = max([(bbox_area, (x_min, y_min, x_max, y_max))], default=(0, (0, 0, 0, 0)))[1]

        
        # 포커싱을 위한 중심점 계산
        focal_point = ((max_subject_bbox[0] + max_subject_bbox[2]) // 2, (max_subject_bbox[1] + max_subject_bbox[3]) // 2)
        
        # 피사체 강조 표시
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.circle(img, focal_point, 10, (0, 0, 255), -1)
    
    # 결과 프레임 출력
    cv2.imshow("Image", img)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
