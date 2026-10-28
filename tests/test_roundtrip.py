import torch

from neuralk_oss import load_model, predict


def test_save_load_predict(tmp_path):
    model = torch.nn.Linear(4, 2)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)

    loaded = load_model(path, torch.nn.Linear(4, 2))
    x = torch.randn(3, 4)

    assert not loaded.training
    torch.testing.assert_close(predict(loaded, x), model(x).detach())
