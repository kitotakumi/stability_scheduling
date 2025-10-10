import csv
import numpy as np
import os
import main_stability_scheduling


def main():
    index = 20
    data = [
        ["makespan", "stability", "changed_sum", "changed_first", "changed_secound", "changed_third", "distance_mean","distance_sum"]
    ]  # fmt:skip
    special_data = [
        ["makespan", "stability", "changed_sum", "changed_first", "changed_secound", "changed_third", "distance_mean", "distance_sum"]
    ]  # fmt:skip
    count = 0
    sum = 0
    while count < index and sum < 100:
        row = main_stability_scheduling.main()
        data.append(list(row))
        if row[0] != 1080:
            special_data.append(list(row))
            count += 1
        sum += 1

    data.append(list(np.mean(data[1:], axis=0)))
    print("統計値", np.mean(data[1:], axis=0))

    with open("output.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print("非収束確率", count / sum)
    print("収束回数", sum - count)
    print("非収束回数", count)
    special_data.append(list(np.mean(data[1:], axis=0)))
    print("収束を除いた統計値", np.mean(special_data[1:], axis=0))


if __name__ == "__main__":
    main()
