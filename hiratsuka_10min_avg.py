import csv
import numpy as np
import pandas as pd

measure_dir_path = "../data/hiratsuka/raw"
avg_10min_dir_path = "../data/hiratsuka/avg_10min"


def main():
    for year in range(2013, 2018):
        for month in range(7, 9):
            for day in range(1, 32):
                avg_data = []
                for hour in range(0, 24):
                    print(year, month, day, hour)
                    try:
                        avg_10min_per_hour = convert_to_avg(
                            year, month, day, hour)
                        avg_data.extend(avg_10min_per_hour)
                    except OSError as e:
                        print(e)

                white_csv(avg_data, year, month, day)


def convert_to_avg(year, month, day, hour):
    data = open_raw_data(year, month, day, hour)
    return calc_10min_avg(data)


def open_raw_data(year, month,  day, hour):
    file_name = "{0}{1:02d}{2:02d}{3:02d}.txt".format(year, month, day, hour)
    file_path = "{0}/{1}/{2}".format(measure_dir_path, year, file_name)

    data = np.genfromtxt(file_path, dtype='float')
    return data[:, 2]


def calc_10min_avg(data):
    avg_data = []
    for i in range(0, 6):
        avg_10min = np.average(data[i * 6000: (i + 1) * 6000])
        avg_data.append(avg_10min)
    return avg_data


def white_csv(data, year, month, day):
    file_name = "{0}{1:02d}{2:02d}.csv".format(year, month, day)
    file_path = "{0}/{1}/{2}".format(avg_10min_dir_path, year, file_name)

    _data = pd.Series(data)
    _data.to_csv(file_path)

if __name__ == "__main__":
    main()
