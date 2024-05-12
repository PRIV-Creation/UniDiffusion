from .unidiffusion_pipeline import UniDiffusionPipeline


class UniDiffusionExportPipeline(UniDiffusionPipeline):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.mode = "export"
        self.default_setup()
        self.build_model()
        # self.build_dataloader()
        # self.set_placeholders()
        # self.prepare_inference()
        self.load_checkpoint()
