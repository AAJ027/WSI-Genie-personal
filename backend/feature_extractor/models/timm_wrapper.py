import torch
import timm

class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50.tv_in1k', 
                 kwargs: dict = {'pretrained': True, "num_classes": 0,}, 
                 pool: bool = False):
        super().__init__()

        assert kwargs.get('pretrained', False), 'only pretrained models are supported'
        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
    
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, list):
            # print(f"shape of output is {len(out), len(out[0]), len(out[0][0]), len(out[0][0][0]), len(out[0][0][0][0])}")
            # assert len(out) == 1, f"shape of output is {len(out), len(out[0]), len(out[0][0]), len(out[0][0][0]), len(out[0][0][0][0])}"
            # assert len(out) == 1, f"shape of output is {out}"
            print("LIST")
            out = out[0]
        if self.pool:
            # print(f"Before squeeze: {out.shape}")
            out = self.pool(out)
            # print(f"After pooling: {out.shape}")
            out= out.squeeze()
            # print(f"after squeeze: {out.shape}")
        return out