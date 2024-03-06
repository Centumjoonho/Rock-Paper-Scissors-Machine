import cv2
import mediapipe as mp
import numpy as np

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# OpenCV 캡처 객체 생성
cap = cv2.VideoCapture(0)

# 화면 크기 확인
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_center = (frame_width // 2, frame_height // 2)

# 이전 프레임에서의 피사체 크기 초기화
prev_subject_area = 0

while True:
    # 프레임 읽기
    success, img = cap.read()
    
    # MediaPipe 처리
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 피사체 리스트 초기화
    subjects = []
    areas = []  # 각 피사체의 영역 크기를 저장할 리스트
    distances_from_center = []  # 각 피사체의 화면 중앙으로부터의 거리를 저장할 리스트
    
    # 각 랜드마크에 대한 처리
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x, y = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
            subjects.append((x, y))
        
        # 바운딩 박스 계산 및 영역 크기 저장
        for subject in subjects:
            x, y = subject
            x_min = min(subjects, key=lambda x: x[0])[0]
            x_max = max(subjects, key=lambda x: x[0])[0]
            y_min = min(subjects, key=lambda x: x[1])[1]
            y_max = max(subjects, key=lambda x: x[1])[1]
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            bbox_area = bbox_width * bbox_height
            areas.append(bbox_area)
            
            # 피사체 중심점과 화면 중앙까지의 거리 계산
            subject_center = (x_min + x_max) // 2, (y_min + y_max) // 2
            distance = np.linalg.norm(np.array(frame_center) - np.array(subject_center))
            distances_from_center.append(distance)
        
        # 우선적으로 트래킹할 피사체 결정
        closest_to_center_index = np.argmin(distances_from_center)
        largest_area_index = np.argmax(areas)
        
        # 화면 중앙에 가장 가까운 피사체의 인덱스로 업데이트
        max_subject_index = closest_to_center_index
        
        # 포커싱을 위한 중심점 계산
        max_subject = subjects[max_subject_index]
        x_min, y_min = max_subject
        x_max = max(subjects, key=lambda x: x[0])[0]
        y_max = max(subjects, key=lambda x: x[1])[1]
        focal_point = ((x_min + x_max) // 2, (y_min + y_max) // 2)
        
        # 피사체 강조 표시
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.circle(img, focal_point, 10, (0, 0, 255), -1)
        
        # 새로운 피사체가 이전에 트래킹된 피사체보다 크다면 트래킹 업데이트
        if max(areas) > prev_subject_area:
            prev_subject_area = max(areas)
        else:
            # 이전에 트래킹된 피사체 유지
            areas[max_subject_index] = prev_subject_area
    
    # 결과 프레임 출력
    cv2.imshow("Image", img)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
