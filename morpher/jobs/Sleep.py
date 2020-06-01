import time

from morpher.jobs import Job


class Sleep(Job):
    def do_execute(self):

        minutes = self.get_input_variables("minutes")
        try:
            minutes = int(minutes)
        except ValueError:
            print("Not a valid integer")
            minutes = 0
        time.sleep(minutes * 60)  # sleep takes time in seconds
        print("Job performed. I slept for {0} minutes.".format(minutes))
