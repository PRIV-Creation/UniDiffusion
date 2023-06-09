class DiffusionTrainer:
    def __init__(self, cfg, training):
        self.cfg = cfg
        self.training = training

        self.build_model()
        if training:
            self.build_dataset()
            self.build_optimizer()
            self.build_scheduler()
            self.build_evaluator()
            self.build_criterion()

    def build_dataset(self):
        pass

    def build_model(self):
        pass

    def build_optimizer(self):
        pass

    def build_scheduler(self):
        pass

    def build_evaluator(self):
        pass

    def build_criterion(self):
        pass

    def train(self):
        pass

    def inference(self):
        pass

    def evaluate(self):
        pass

