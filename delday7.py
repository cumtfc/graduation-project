import time
import os

file = open('./data/round1_ijcai_18_train_20180301.txt')
if os.path.isfile('./data/round2_train2.txt'):
    os.remove('./data/round2_train2.txt')

file2 = open('./data/round2_train2.txt', 'w')
cnt = 0
cnt_all = 0
for line in file:
    line_split = line.split(sep=" ")
    time_stamp = line_split[16]
    if time_stamp.isdigit():
        st = time.localtime(float(time_stamp))
        day = time.strftime('%d', st)
        cnt_all += 1
        print(day)
        if day != '07':
            file2.write(line)
            cnt += 1

file2.flush()
file.close()
file2.close()
print(cnt, "/", cnt_all)
