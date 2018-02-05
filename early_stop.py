class EarlyStop:
    def __init__(self, tolerance):
        self.epoch = 0
        self.loss_list = []
        self.tolerance = tolerance

    def new_loss(self, loss):
        self.loss_list.append(loss)

    def if_stop(self):
        if len(self.loss_list) < self.tolerance:
            return False
        tmp_list = self.loss_list[-self.tolerance:]
        if sorted(tmp_list) == tmp_list:
            return True
        else:
            return False
