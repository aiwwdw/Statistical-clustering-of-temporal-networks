import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

# 파일 경로 및 설정
setting = '100_10_0p75'
file_paths = [
    f'plot/{setting}/{setting}_boxplot_global_average.png',          # Ours - Random
    f'plot/{setting}/{setting}_boxplot_kmeans_global_average.png',   # Ours - K-means
    f'plot/{setting}/{setting}_boxplot_prior_random_global_average.png', # Previous - Random
    f'plot/{setting}/{setting}_boxplot_prior_kmeans_global_average.png'  # Previous - K-means
]

# Figure와 GridSpec을 사용하여 subplot 배치
fig = plt.figure(figsize=(12, 8))  # 가로 12, 세로 8로 설정
gs = GridSpec(3, 3, figure=fig, width_ratios=[0.2, 10, 10], height_ratios=[0.2, 10, 10])  # 1행과 1열 크기 최소화

# 제목 텍스트 추가
fig.text(0.5, 0.93, f"Comparison in {setting}", ha='center', fontsize=16)

# 1행과 1열에 제목 텍스트 추가
ax1 = fig.add_subplot(gs[0, 1])
ax1.set_title("Random", fontsize=14, pad=0)
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 2])
ax2.set_title("K-means", fontsize=14, pad=0)
ax2.axis('off')

ax3 = fig.add_subplot(gs[1, 0])
ax3.annotate("Ours", xy=(0.5, 0.5), xycoords='axes fraction', va='center', ha='center', fontsize=14, rotation=90)
ax3.axis('off')

ax4 = fig.add_subplot(gs[2, 0])
ax4.annotate("Previous", xy=(0.5, 0.5), xycoords='axes fraction', va='center', ha='center', fontsize=14, rotation=90)
ax4.axis('off')

# 이미지 표시할 subplot에 파일 추가
for i, file_path in enumerate(file_paths):
    img = mpimg.imread(file_path)  # 이미지 읽기
    row, col = divmod(i, 2)        # 2x2 위치 계산 (1행과 1열 제외)
    ax = fig.add_subplot(gs[row + 1, col + 1])  # 이미지 배치
    ax.imshow(img)                  # 이미지 표시
    ax.axis('off')                  # 축 숨기기

# 여백 줄이기
plt.subplots_adjust(left=0.1, right=0.9, top=0.88, bottom=0.1, wspace=0.05, hspace=0.05)  # 간격 조정

# 파일로 저장
plt.savefig(f'plot/overall/output_{setting}.png', dpi=300, bbox_inches='tight')  # 여백 제거하여 저장
plt.show()