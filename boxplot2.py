import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



setting = '100_5_0p9'



data_frames = []
labels = ['low-', 'low+', 'medium-', 'medium+', 'medium w/ affiliation']
est_types = ['', 'kmeans_','prior_random_','prior_kmeans_']

file_paths = [
    f'output/{setting}/result_low_minus_{setting}.csv',
    f'output/{setting}/result_low_plus_{setting}.csv',
    f'output/{setting}/result_medium_minus_{setting}.csv',
    f'output/{setting}/result_medium_plus_{setting}.csv',
    f'output/{setting}/result_medium_with_affiliation_{setting}.csv'
]

for est_type in est_types:
    for file_path, label in zip(file_paths, labels):
        global_ari_str = f'{est_type}global_ari'
        average_ari_str = f'{est_type}average_ari'

        data = pd.read_csv(file_path)
        melted_data = data.melt(value_vars=[global_ari_str, average_ari_str], 
                                var_name='ARI_Type', 
                                value_name='ARI_Value')
        melted_data['label'] = label
        data_frames.append(melted_data)

    # 데이터프레임들을 합칩니다.
    combined_data = pd.concat(data_frames)

    # 각 ARI_Type에 대해 개별 boxplot 생성
    plt.figure(figsize=(15, 10))

    positions_global = [i - 0.2 for i in range(len(labels))]
    positions_average = [i + 0.2 for i in range(len(labels))]

    # 평균선 스타일 정의 (굵게 설정)
    meanprops = {'color': 'black', 'linestyle': '-', 'linewidth': 2}

    # Boxplot 생성 (범례 없이)
    sns.boxplot(x='label', y='ARI_Value', data=combined_data[combined_data['ARI_Type'] == global_ari_str],
                color='white', width=0.32, linewidth=1.5, 
                positions=positions_global, meanline=True, showmeans=True, meanprops=meanprops)

    sns.boxplot(x='label', y='ARI_Value', data=combined_data[combined_data['ARI_Type'] == average_ari_str],
                color='gray', width=0.32, linewidth=1.5, 
                positions=positions_average, meanline=True, showmeans=True, meanprops=meanprops)

    # 'medium+'와 'medium w/ affiliation' 사이에 수직선 그리기
    plt.axvline(x=3.5, color='black', linestyle='-', linewidth=1.5)

    # 범례를 수동으로 추가
    global_legend = plt.Line2D([0], [0], color="white", lw=4, label='Global ARI')
    average_legend = plt.Line2D([0], [0], color="gray", lw=4, label='Average ARI')
    plt.legend(handles=[global_legend, average_legend], title='ARI Type')

    # 플롯 세부사항 설정
    plt.title(f'Boxplot of Global ARI (White) and Average ARI (Gray) in {est_type}')
    plt.ylabel('Adjusted Rand Index')
    plt.xticks(range(len(labels)), labels)

    # 플롯을 파일로 저장합니다.
    output_file_path = f'plot/{setting}/{setting}_boxplot_{est_type}global_average.png'
    plt.savefig(output_file_path)

    # 플롯을 표시합니다.
    plt.show()