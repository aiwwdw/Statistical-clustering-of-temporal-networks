import torch

# (2, 2, 100, 100) 크기의 텐서 생성
tensor = torch.randn(2, 2, 100, 100)

# (100, 100) 부분의 upper triangular matrix의 non-diagonal 요소들 추출
indices = torch.triu_indices(100, 100, offset=1)

# upper triangular matrix의 non-diagonal 요소들 추출 및 합계 계산
upper_triangular_elements = tensor[:, :, indices[0], indices[1]]
final_sum = upper_triangular_elements.sum()

print(final_sum)