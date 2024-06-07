import time


class Timer:

    def __init__(self, record: dict[str: list]={}):
        self.record = record
    
    def get_record(self) -> dict:
        return self.record
    
    def restore_record(self, record: dict[str: list]) -> None:
        self.record.update(record)

    def start_record(self, name: str):
        now = time.time()
        self.record[name] = [now, now]
    
    def end_record(self, name: str):
        now = time.time()
        if name in self.record:
            self.record[name][1] = now
        else:
            self.record[name] = [now, now]

    def get_time(self, name):
        if name in self.record:
            return self.record[name][1] - self.record[name][0]
        else:
            return 0.0
    
    def get_total_time(self):
        total_time = 0
        for name in self.record.keys():
            total_time += self.get_time(name)
        return total_time
