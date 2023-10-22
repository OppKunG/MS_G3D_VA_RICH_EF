from datetime import datetime


def print_time():
    current_datetime = datetime.now()
    current_date_time = current_datetime.strftime("%m/%d/%Y, %H:%M:%S")

    return current_date_time
