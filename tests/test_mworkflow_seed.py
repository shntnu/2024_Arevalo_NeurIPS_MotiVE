import json
from types import SimpleNamespace

import torch

import mworkflow


def test_init_seeds_model_construction(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "leave_out": "target",
                "target_type": "crispr",
                "graph_type": "st_expanded",
                "neg_ratio": 100,
            }
        )
    )
    loader = SimpleNamespace(loader=SimpleNamespace(data=object()))
    monkeypatch.setattr(mworkflow, "get_loaders", lambda *_: (loader, loader, loader))
    draws = []

    class Model:
        def to(self, _device):
            return self

    def create_model(_config, _data):
        draws.append(torch.rand(4))
        return Model()

    monkeypatch.setattr(mworkflow, "create_model", create_model)
    mworkflow.init(config_path)
    mworkflow.init(config_path)
    torch.testing.assert_close(draws[0], draws[1], rtol=0, atol=0)
