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
    df = open_raw_dataframe(year, month, day, hour)
    return calc_10min_avg(df)


def open_raw_dataframe(year, month,  day, hour):
    file_name = "{0}{1:02d}{2:02d}{3:02d}.txt".format(year, month, day, hour)
    file_path = "{0}/{1}/{2}".format(measure_dir_path, year, file_name)

    data = np.genfromtxt(file_path, dtype='float')
    return data[:, 2:3 + 1]


def calc_10min_avg(data):
    vel_abs = data[:, 0]  # 風速
    rad = np.radians(data[:, 1])  # 風向[rad] 北から時計回り
    sin = np.sin(rad)  # 風向東西成分（東+）
    cos = np.cos(rad)  # 風向南北成分（北+）
    vel_we = -1 * vel_abs * sin  # 風東西成分（東+）風向は風が"吹いて来る"方向
    vel_ns = -1 * vel_abs * cos  # 風南北成分（北+）

    avg_data = []
    for i in range(0, 6):
        avg_vel_we = np.average(vel_we[i * 6000: (i + 1) * 6000])
        avg_vel_ns = np.average(vel_ns[i * 6000: (i + 1) * 6000])
        avg_data.append([avg_vel_ns, avg_vel_we])
    # print(avg_data)
    return avg_data


def white_csv(data, year, month, day):
    file_name = "{0}{1:02d}{2:02d}.csv".format(year, month, day)
    file_path = "{0}/{1}/{2}".format(avg_10min_dir_path, year, file_name)

    _data = pd.DataFrame(data, columns=["VGRD", "UGRD"])
    _data.to_csv(file_path)

if __name__ == "__main__":
    main()
