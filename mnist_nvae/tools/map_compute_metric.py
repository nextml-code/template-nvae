from ignite.metrics.metric import Metric


class MapComputeMetric(Metric):
    def __init__(self, map_fn, compute_fn):
        super().__init__(output_transform=map_fn)
        self.compute_fn = compute_fn

    def reset(self):
        self.__batches = []

    def update(self, batch):
        self.__batches.append(batch)

    def compute(self):
        return self.compute_fn(self.__batches)
