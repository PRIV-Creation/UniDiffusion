from .unidiffusion_pipeline import UniDiffusionPipeline


class UniDiffusionEvaluationPipeline(UniDiffusionPipeline):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.training = False
        self.default_setup()
        self.build_model()
        self.build_dataloader()
        self.set_placeholders()
        self.build_evaluator()
        self.prepare_inference(prepare_evaluator=True)
        self.load_checkpoint()
