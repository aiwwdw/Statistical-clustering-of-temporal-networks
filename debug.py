import torch
import torch.backends.cuda
print(f"MPS 장치를 지원하도록 build가 되었는가? {torch.backends.cuda.is_built()}")
print(f"MPS 장치가 사용 가능한가? {torch.backends.cuda.is_available()}") 