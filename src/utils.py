from datetime import datetime

class Run:
    def __init__(self):
        self.timetag = datetime.now().strftime("%Y%m%d-%H%M%S")

    def get_run_dir(self, tag):
        return "runs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/{tag}/"
