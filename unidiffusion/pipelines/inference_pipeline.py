from .unidiffusion_pipeline import UniDiffusionPipeline


class UniDiffusionInferencePipeline(UniDiffusionPipeline):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.mode = "inference"
        self.default_setup()
        self.build_model()
        self.build_dataloader()
        self.set_placeholders()
        self.prepare_inference()
        self.load_checkpoint()
