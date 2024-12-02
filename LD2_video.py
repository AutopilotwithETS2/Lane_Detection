import cv2
import numpy as np

# Sobel 필터 적용 함수
def sobel_xy(img, orient='x', thresh=(20, 100)):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    else:
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel)) if np.max(abs_sobel) != 0 else np.zeros_like(img)
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255
    return binary_output

# 프레임을 전처리하고 흰색에 가까운 색만 남기기
def preprocess_frame(frame):
    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러 적용
    blurred_frame = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 소벨 필터 적용
    sobel_x = sobel_xy(blurred_frame, 'x', (35, 100))
    sobel_y = sobel_xy(blurred_frame, 'y', (30, 255))

    # 그라디언트 결합
    grad_combine = np.zeros_like(sobel_x).astype(np.uint8)
    grad_combine[((sobel_x > 1) & (sobel_y > 1))] = 255

    # Canny 엣지 검출
    edges = cv2.Canny(blurred_frame, 50, 150)

    # Sobel과 Canny 결합
    combined = cv2.bitwise_or(grad_combine, edges)

    return combined

# 세로선을 검출하는 함수
def detect_vertical_lines(frame):
    processed_frame = preprocess_frame(frame)
    
    height, width = processed_frame.shape
    
    # ROI 영역 설정 (사다리꼴 형태)
    mask = np.zeros_like(processed_frame)
    roi_vertices = np.array([[
        (350, 510),  # 왼쪽 아래
        (350, 300),  # 왼쪽 위
        (680, 300),  # 오른쪽 위
        (width, 510) # 오른쪽 아래
    ]], dtype=np.int32)
    
    # ROI 영역 시각화
    cv2.polylines(frame, [roi_vertices], isClosed=True, color=(0, 0, 255), thickness=3)

    # ROI 마스크를 적용하여 ROI 영역 내의 엣지만 남기기
    cv2.fillPoly(mask, [roi_vertices], 255)
    masked_edges = cv2.bitwise_and(processed_frame, mask)

    # 검출된 엣지를 확인하기 위한 디버그용 시각화
    cv2.imshow('Masked Edges', masked_edges)

    # 허프 변환을 이용하여 직선 검출
    lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=30, maxLineGap=20)
    
    # 검출된 선을 그릴 이미지 생성
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else np.inf
            
            # 기울기 절대값이 작은 경우 가로선으로 간주하고 제외
            if abs(slope) < 0.1:
                continue
            
            # 세로선으로 인식된 선을 그리기
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    # 원본 이미지와 라인 이미지를 합침
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    
    return result

# 영상을 재생하고 선 검출 및 ROI 영역을 표시하는 함수
def show_video_processing(video_path):
    cv2.namedWindow('Processed Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Processed Video', 800, 600)

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임에서 차선 인식 및 ROI 시각화
        lane_frame = detect_vertical_lines(frame)
        
        # 화면에 표시
        cv2.imshow('Processed Video', lane_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__
    
    
    
    
    
    
    
    q
    
    th = "C:\\Users\\girookim\\Desktop\\Autopiliot\\test2.mp4"
    video_path2 = "C:\\Users\\girookim\\Desktop\\Autopiliot\\test.mp4"
    show_video_processing(video_path2)
