from transformers import LlamaConfig


class LoKILlamaConfig(LlamaConfig):
    def __init__(self, target_neurons=None, **kwargs):
        self.target_neurons = target_neurons  # 添加新参数
        super().__init__(**kwargs)

