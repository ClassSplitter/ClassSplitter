import time


class Timer:

    def __init__(self):
        self.init_time = time.time()
        self.time_records = []
        self.period_names = []

    def record_past(self, name):
        self.time_records.append(time.time())
        self.period_names.append(name)

    def restart(self):
        this_init = time.time()
        diff = this_init - self.time_records[-1]
        for i in range(len(self.time_records)):
            self.time_records[i] = self.time_records[i] + diff
        self.init_time = self.init_time + diff

    def get_total_time(self):
        return self.time_records[-1] - self.init_time

    def get_time(self, label):
        if label in self.period_names:
            index = self.period_names.index(label)
            if index >= 1:
                return self.time_records[index] - self.time_records[index - 1]
            else:
                return self.time_records[0] - self.init_time
        return 0
