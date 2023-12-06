from .unidiffusion_pipeline import UniDiffusionPipeline


class UniDiffusionTrainingPipeline(UniDiffusionPipeline):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.mode = "training"
        self.default_setup()
        self.build_model()
        self.prepare_db()
        self.build_dataloader()
        self.prepare_null_text()
        self.set_placeholders()
        self.build_optimizer()
        self.build_scheduler()
        self.build_evaluator()
        self.prepare_training()
        self.load_checkpoint()
        self.print_training_state()
