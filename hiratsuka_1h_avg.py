import csv
import numpy as np
import pandas as pd

avg_10min_path = "../data/hiratsuka/avg_10min"
avg_1h_dir_path = "../data/hiratsuka/avg_1h"


def main():
    for year in range(2013, 2018):
        for month in range(7, 9):
            for day in range(1, 32):
                print(year, month, day)
                try:
                    avg_1h = convert_to_avg(year, month, day)
                    white_csv(avg_1h, year, month, day)
                except IndexError as e:
                    print(e)


def convert_to_avg(year, month, day):
    data = open_10min_data(year, month, day)
    return calc_1h_avg(data)


def open_10min_data(year, month,  day):
    file_name = "{0}{1:02d}{2:02d}.csv".format(year, month, day)
    file_path = "{0}/{1}/{2}".format(avg_10min_path, year, file_name)

    data = np.genfromtxt(file_path, delimiter=",", dtype='float')
    return data[:, 1]


def calc_1h_avg(data):
    avg_data = []
    for i in range(0, 24):
        avg_1h = np.average(data[i * 6: (i + 1) * 6])
        avg_data.append(avg_1h)
    return avg_data


def white_csv(data, year, month, day):
    file_name = "{0}{1:02d}{2:02d}.csv".format(year, month, day)
    file_path = "{0}/{1}/{2}".format(avg_1h_dir_path, year, file_name)

    _data = pd.Series(data)
    _data.to_csv(file_path)

if __name__ == "__main__":
    main()
