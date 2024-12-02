import cv2
import numpy as np

# 빨간 점 표시 & 녹색 선 표시
def draw_detected_lines(frame, lines, color=(0, 255, 0)):  # 기본 색상은 초록색
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if 45 <= abs(angle) <= 90 and line_length >= 100:
                num_points = 2
                t = np.linspace(0, 1, num_points)
                x_points = (1 - t) * x1 + t * x2
                y_points = (1 - t) * y1 + t * y2
                points = np.stack((x_points, y_points), axis=-1).astype(int)

                for point in points:
                    cv2.circle(frame, tuple(point), 10, (0, 0, 255), -1)

                curve = np.polyfit(x_points, y_points, 2)
                curve_x = np.linspace(x_points[0], x_points[-1], num_points * 5)
                curve_y = np.polyval(curve, curve_x)
                curve_points = np.stack((curve_x, curve_y), axis=-1).astype(int)

                cv2.polylines(frame, [curve_points], isClosed=False, color=color, thickness=2)

def sobel_xy(img, orient='x', thresh=(20, 100)):
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    else:
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel)) if np.max(abs_sobel) != 0 else np.zeros_like(img)
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255
    return binary_output

def show_video_processing(video_path):
    cv2.namedWindow('Processed Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Processed Video', 800, 600)

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 처리
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 가우시안 블러 적용
        temp = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)

        # 소벨 필터 적용
        sobel_x = sobel_xy(temp, 'x', (35, 100))
        sobel_y = sobel_xy(temp, 'y', (30, 255))

        # 그라디언트 결합
        grad_combine = np.zeros_like(sobel_x).astype(np.uint8)
        grad_combine[((sobel_x > 1) & (sobel_y > 1))] = 255

        # Canny 엣지 검출
        canny_edges = cv2.Canny(temp, 50, 150)

        # ROI 지정
        width, height = sobel_x.shape
        vertices = np.array([[(350, 510), (350, 300), (680, 300), (width, 510)]], dtype=np.int32)
        mask = np.zeros_like(sobel_x)
        cv2.fillPoly(mask, vertices, 255)

        # ROI 영역에서만 edge 검출
        masked_sobel_edges = cv2.bitwise_and(grad_combine, mask)
        masked_canny_edges = cv2.bitwise_and(canny_edges, mask)

        # 선분 찾기
        lines_sobel = cv2.HoughLinesP(masked_sobel_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)
        lines_canny = cv2.HoughLinesP(masked_canny_edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)

        draw_detected_lines(frame, lines_sobel, color=(0, 255, 0))  # 소벨 선은 초록색
        draw_detected_lines(frame, lines_canny, color=(255, 0, 0))  # Canny 선은 파란색

        # 사다리꼴 영역을 하늘색으로 표시
        cv2.polylines(frame, vertices, isClosed=True, color=(135, 206, 250), thickness=2)

        # 화면 출력
        cv2.imshow('Processed Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_night = "C:\\Users\\girookim\\Desktop\\Autopiliot\\test.mp4"
    video_rain = "C:\\Users\\girookim\\Desktop\\Autopiliot\\test4.mp4"
    video_top_view = "C:\\Users\\girookim\\Desktop\\Autopiliot\\test3.mp4"
    show_video_processing(video_rain)
