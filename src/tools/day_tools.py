from datetime import datetime


def getTimeStamp(minute=True):
    if minute:
        return datetime.now().strftime("%y%m%d_%H_%M")
    else:
        return datetime.now().strftime("%y%m%d")
