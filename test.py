import pandas as pd

# data_path = "C:/Users/Administrator/Desktop/ssim.csv"
#
# df = pd.read_csv(data_path)
#
# print(df.describe())
#
# print(df.query('ssim >= 0.96')['image'])


train1_df = pd.read_csv("yolov8/runs/detect/train/results.csv")
train2_df = pd.read_csv("yolov8/runs/detect/train2/results.csv")
train3_df = pd.read_csv("yolov8/runs/detect/train3/results.csv")
train4_df = pd.read_csv("yolov8/runs/detect/train4/results.csv")

c = train1_df.columns

# for cc in c:
#     print(cc)

t1 = train1_df[c[7]].tolist()
t2 = train2_df[c[7]].tolist()
t3 = train3_df[c[7]].tolist()
t4 = train4_df[c[7]].tolist()


def get_avg(list_):
    return sum(list_) / len(list_)


print("t1"+c[7]+":", get_avg(t1))
print("t2"+c[7]+":", get_avg(t2))
print("t3"+c[7]+":", get_avg(t3))
print("t4"+c[7]+":", get_avg(t4))
