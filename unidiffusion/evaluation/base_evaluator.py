class BaseEvaluator:
    name = None

    def before_train(self, dataset, accelerator):
        pass
