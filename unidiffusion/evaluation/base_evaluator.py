class BaseEvaluator:
    name = None

    def before_train(self, dataset, accelerator):
        pass

    # def update(self, **kwargs):
    #     pass

    def update_by_evaluator_name(self, name, **kwargs):
        if self.name == name:
            self.update(**kwargs)
