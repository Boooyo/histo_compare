import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class HistCompMethod(Enum):
    CORREL = cv2.HISTCMP_CORREL
    CHISQR = cv2.HISTCMP_CHISQR
    INTERSECT = cv2.HISTCMP_INTERSECT
    BHATTACHARYYA = cv2.HISTCMP_BHATTACHARYYA

def calculate_normalized_histogram(image):
    """이미지의 히스토그램을 계산하고 정규화합니다."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # HSV로 변환
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256]) # H, S 채널 히스토그램 계산
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX) # 0~1로 정규화
    return hist

def compare_histograms(base_hist, comp_hist, method):
    """지정된 방법을 사용하여 두 히스토그램을 비교합니다."""
    result = cv2.compareHist(base_hist, comp_hist, method.value)
    if method == HistCompMethod.INTERSECT: # 교차 분석인 경우
        result /= np.sum(base_hist) # 비교 대상으로 나누어 1로 정규화
    return result

def plot_images(images, titles):
    """이미지 목록과 그 제목으로 이미지를 표시합니다."""
    plt.figure(figsize=(10, 2.5))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image[:, :, ::-1])
        plt.title(title)
        plt.axis('off')
    plt.show()

def main():
    # 이미지 로드
    image_paths = ['../img/taekwonv1.jpg', '../img/taekwonv2.jpg', '../img/taekwonv3.jpg', '../img/dr_ochanomizu.jpg']
    images = [cv2.imread(path) for path in image_paths]

    # 히스토그램 계산 및 정규화
    hists = [calculate_normalized_histogram(img) for img in images]

    # 히스토그램 비교
    methods = [HistCompMethod.CORREL, HistCompMethod.CHISQR, HistCompMethod.INTERSECT, HistCompMethod.BHATTACHARYYA]
    for method in methods:
        print(f'{method.name:<15}', end='')
        for i, hist in enumerate(hists):
            similarity = compare_histograms(hists[0], hist, method)
            print(f"img{i+1}:{similarity:7.2f}", end='\t')
        print()

    # 이미지 표시
    plot_images(images, ['img1', 'img2', 'img3', 'img4'])

if __name__ == "__main__":
    main()
