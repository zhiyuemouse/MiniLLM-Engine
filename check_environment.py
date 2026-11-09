import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "PyTorch 无法使用 CUDA"

print("GPU:", torch.cuda.get_device_name(0))

x = torch.tensor([[1., 2.], [3., 4.]], device="cuda")
actual = (x @ x).cpu()
expected = torch.tensor([[7., 10.], [15., 22.]])

torch.testing.assert_close(actual, expected)
print("GPU 矩阵乘法结果:", actual.tolist())
print("PASS")