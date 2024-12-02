import cv2
import numpy as np
import pygetwindow as gw
from PIL import ImageGrab

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
    blurred_frame = cv2.GaussianBlur(gray, (5, 5), 0)  # 가우시안 블러 적용

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

    return combined, edges  # Canny 엣지도 반환

# 세로선을 검출하는 함수
def detect_vertical_lines(frame, line_thickness=3, alpha=0.8):
    processed_frame, edges = preprocess_frame(frame)

    height, width = processed_frame.shape

    # ROI 영역 설정 (사다리꼴 형태)
    mask = np.zeros_like(processed_frame)
    roi_vertices = np.array([[ 
        (380, 510),  # 왼쪽 아래
        (380, 300),  # 왼쪽 위
        (660, 300),  # 오른쪽 위
        (720, 510)   # 오른쪽 아래
    ]], dtype=np.int32)

    # ROI 영역 시각화 (원본 프레임에 사다리꼴을 빨간색으로 표시)
    cv2.polylines(frame, [roi_vertices], isClosed=True, color=(0, 0, 255), thickness=3)

    # ROI 마스크를 적용하여 ROI 영역 내의 엣지만 남기기
    cv2.fillPoly(mask, [roi_vertices], 255)
    masked_edges = cv2.bitwise_and(processed_frame, mask)

    # 점 설정
    apex = (530, 300)
    left_bottom = (380, 510)
    right_bottom = (720, 510)

    # 두 개의 직선 그리기 (Cyan 색상)
    overlay = frame.copy()
    cv2.line(overlay, apex, left_bottom, (255, 255, 0), thickness=line_thickness)  # 첫 번째 직선 (Cyan)
    cv2.line(overlay, apex, right_bottom, (255, 255, 0), thickness=line_thickness)  # 두 번째 직선 (Cyan)

    # 투명도 효과를 위한 마스크 생성
    alpha_mask = overlay.astype(float) * alpha + frame.astype(float) * (1 - alpha)
    frame = alpha_mask.astype(np.uint8)

    # Canny 엣지 검출과 Cyan 선의 겹치는 부분 검출
    cyan_mask = np.zeros_like(edges)
    cv2.line(cyan_mask, apex, left_bottom, 255, thickness=line_thickness)
    cv2.line(cyan_mask, apex, right_bottom, 255, thickness=line_thickness)

    # 겹치는 부분 마스크 생성
    overlap_mask = cv2.bitwise_and(edges, cyan_mask)

    # 겹치는 부분에 초록색 오버레이
    green_overlay = np.zeros_like(frame)  # 초록색 오버레이를 위한 빈 이미지
    for pixel in np.argwhere(overlap_mask == 255):
        y, x = pixel
        cv2.circle(green_overlay, (x, y), 2, (0, 255, 0), thickness=-1)  # 겹치는 부분에 초록색 원

    # 검출된 엣지를 확인하기 위한 디버그용 시각화
    cv2.imshow('Masked Edges', masked_edges)

    return frame, green_overlay  # 원본 프레임과 초록색 오버레이 반환

# 실시간 화면을 보여주는 함수
def show_screen(window_title):
    cv2.namedWindow('Real-Time Screen', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Real-Time Screen', 1024, 768)  # 게임 창 크기와 맞추기

    while True:
        try:
            window = gw.getWindowsWithTitle(window_title)[0]
            left, top, right, bottom = window.left, window.top, window.right, window.bottom
            
            # 이미지 캡처
            screen = np.array(ImageGrab.grab(bbox=(left, top, right, bottom)))
            frame = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

            # 프레임에서 차선 인식 및 ROI 시각화
            lane_frame, green_overlay = detect_vertical_lines(frame, line_thickness=20, alpha=0.4)  # 두께를 20으로 설정, 투명도 40%
            
            # 화면에 표시
            cv2.imshow('Real-Time Screen', lane_frame)

            # 원본 화면에 ROI와 초록색 오버레이 표시
            overlay_frame = cv2.addWeighted(frame, 1.0, green_overlay, 1.0, 0)
            cv2.imshow('Overlay Window', overlay_frame)

        except IndexError:
            print(f"'{window_title}' 창을 찾을 수 없습니다.")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    window_title = "Euro Truck Simulator 2"  # 게임 창의 제목
    show_screen(window_title)
