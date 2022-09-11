# -*- coding: utf-8 -*-
# @Time : 2022/8/24 0:46
# @Author : LuoJiahuan
# @Email : luojiahuan001@gmail.com
# @File : merge_result.py

if __name__ == '__main__':
    cls = open("exp/pre_cls.csv").readlines()
    reg = open("exp/pre_reg.csv").readlines()
    with open("exp/pre.csv", "w") as f:
        for line in cls:
            f.write(line)
        for line in reg:
            id = int(line[0])
            line = str(int(id + 8)) + line[1:]
            f.write(line)
        f.close()