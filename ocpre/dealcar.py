import os
import re
import math
import lmdb
import torch
import random
import pickle
import shutil
import itertools
import collections
import numpy as np
import matplotlib.pyplot as plt

from ase.atoms import Atoms
from vaspvis import standard
from ase.build import add_vacuum
from ase.calculators.emt import EMT
from ase.optimize import LBFGS, BFGS
from ase.visualize.plot import plot_atoms
from ase.io import write, read, lammpsdata
from ocpmodels.datasets import LmdbDataset
from ocpmodels.preprocessing import AtomsToGraphs
from ase.constraints import FixAtoms, FixCartesian
from ase.build import add_adsorbate, molecule, fcc111

def write_atoms(filename, atoms, format="vasp"):
    # 该函数的作用是修正 ase 的 write 函数，可以输出固定特定原子自由度的约束
    write(filename, atoms, format="vasp")
    if not any(
        isinstance(constraint, FixCartesian)
        for constraint in atoms.constraints
    ):
        return "write atoms only"
    cons = atoms.constraints
    # read part
    data = read_message(filename)  # 读取文件
    mbox, mnum, mdata, info = deal_message(data)  # 解析poscar
    for con in cons:
        if type(con) == FixCartesian:
            # deal part
            for i in con.a:
                string_list = info[i]
                result_list = [
                    string_list[i] if con.mask[i] else "F"
                    for i in range(len(con.mask))
                ]
                info[i] = result_list
    # write part
    newdata = joint(data, mbox, mnum, mdata, info)
    write_message(newdata, filename=filename)  # 输出poscar
    write(filename, read(filename), format="vasp")  # 读写检验
    return "write FixCartesian"


def out_poscar(input, path="New_Folder"):  # sourcery skip: extract-method
    # 如果文件夹已经存在，则删除该文件夹及其中的内容
    if os.path.exists(path):
        shutil.rmtree(path)
    # 创建新的文件夹
    os.makedirs(path)
    if isinstance(input, list):
        print("input is list")
        all_model = input
        for i in range(len(all_model)):
            atoms = all_model[i]
            filepath = os.path.join(".", path, str(int(i + 1e3)))
            os.makedirs(filepath)
            filename = os.path.join(filepath, "POSCAR")
            write_atoms(filename, atoms, format="vasp")
    elif isinstance(input, dict):
        print("input is dict")
        all_model = input
        name = all_model.keys()
        for key in name:
            atoms = all_model[key]
            filepath = os.path.join(".", path, key)
            os.makedirs(filepath)
            filename = os.path.join(filepath, "POSCAR")
            write_atoms(filename, atoms, format="vasp")
    elif isinstance(input, Atoms):
        print("input is Atoms")
        atoms = input
        filepath = path
        filename = os.path.join(filepath, "POSCAR")
        write_atoms(filename, atoms, format="vasp")
    else:
        raise ValueError("input must be a list or dict or Atoms")

def out_car_list(input, path="New_Folder"):
    # 这个函数中嵌套了out_poscar，所以实际输出是嵌套文件夹
    # 如果文件夹已经存在，则删除该文件夹及其中的内容
    if os.path.exists(path):
        shutil.rmtree(path)
    # 创建新的文件夹
    os.makedirs(path)
    if isinstance(input, list):
        for i in range(len(input)):
            filepath = os.path.join(".", path, str(int(i + 1e3)))
            os.makedirs(filepath)
            out_poscar(input[i], path=filepath)
    else:
        raise ValueError("input must be a list")

def get_dos_data(
    dos_folder, atom_num=6, orbital_list=list(range(9)), erange=[-8, 8],no_plot=True
):  
    # print(dos_folder)
    # 该函数可以获得指定文件夹的dos信息
    # 默认是POSCAR的第七个原子（atom_num=6），9个轨道，能量范围[-11,11]eV
    dos_data = standard.dos_atom_orbitals(
        atom_orbital_dict={atom_num: orbital_list},
        folder=dos_folder,
        erange=[erange[0]-1, erange[1]+1],
        total=False,
        save=False,
        figsize=[5, 3],
    )
    figure, axe = dos_data
    if no_plot:
        plt.close(figure)  # 关闭图像显示，不在命令行中展示
    x0 = axe.lines[0].get_data()[0]
    x = x0[(x0 > erange[0]) & (x0 <= erange[1])]
    dos_data = np.zeros((len(x), len(orbital_list) + 1))
    dos_data[:, 0] = x
    for i in range(len(orbital_list)):
        y = axe.lines[i].get_data()[-1]
        y = y[(x0 > erange[0]) & (x0 <= erange[1])]
        dos_data[:, i + 1] = y
    return dos_data

def get_energy(OSZICAR_file, num=-1):
    # 该函数可以获得OSZICAR文件中的能量值，如果存在的话
    lines = []
    with open(OSZICAR_file, "r") as oszicar:
        # 逐行读取文件内容
        lines.extend(line for line in oszicar if " F= " in line)
    lastline = lines[num]
    if match := re.search(r"E0= ([-+]?\d*\.\d+E[+-]?\d+)", lastline):
        e0_value = match.group(1)
        print("E0 value:", num, "F  ", e0_value)
    else:
        print("No E0 value found.")
    e0_value = float(e0_value)
    return e0_value

def read_one_car(path, car="CONTCAR"):
    """
    可以从path中读取指定类型的文件，path是计算文件夹
    支持文件类型： car='CONTCAR' or 'POSCAR' or 'OSZICAR' or 'DOSCAR
    
    Args:
        path (str): Path can be a ① calculated folder, 
        or a ② parent folder of a list folder
        car (str, optional): car='CONTCAR' or 'POSCAR' or 'OSZICAR' or 'DOSCAR.
        Defaults to "CONTCAR".
    Returns:
        'CONTCAR' or 'POSCAR' return atoms of type ase.atom
        'OSZICAR' return Energy of type float.64
        'DOSCAR' return array of type np.array
    """

    if type(car) == list:
        num = car[1]
        car = car[0]
    else:
        num = -1  
    filename = os.path.join(path, car)
    if os.path.isfile(filename):
        if "OSZICAR" in car:
            atom = (
                get_energy(filename, num=num)
                if num != None
                else get_energy(filename)
            )  # 可以指定要读取OSZICAR中的第几个离子步的能量
        elif "DOSCAR" in car:
            atom = get_dos_data(path)
        else:
            atom = read(filename, format="vasp")
    print(filename)
    return (atom)

def read_car(path, car="CONTCAR"):
    """
    可以从path中读取指定类型的文件，path是列表文件夹的父文件夹
    返回值是一个列表和一个字典，当子文件夹的名字是数字时，会返回升序排列好的列表和字典
    支持文件类型： car='CONTCAR' or 'POSCAR' or 'OSZICAR' or 'DOSCAR
    
    Args:
        path (str): Path can be a ① calculated folder, 
        or a ② parent folder of a list folder
        car (str, optional): car='CONTCAR' or 'POSCAR' or 'OSZICAR' or 'DOSCAR.
        Defaults to "CONTCAR".
        
    Returns:
        'CONTCAR' or 'POSCAR' return atoms of type ase.atom
        'OSZICAR' return Energy of type float.64
        'DOSCAR' return array of type np.array
    """

    # if type(car) == list:
    #     num = car[1]
    #     car = car[0]
    # else:
    #     num = -1
    all_list = []
    all_dict = {}
    subfolders = [
        f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))
    ]
    for subfolder in subfolders:
        folder_path = os.path.join(path, subfolder)
        atom = read_one_car(folder_path, car)
        # filename = os.path.join(folder_path, car)
        # if os.path.isfile(filename):
        #     if "OSZICAR" in car:
        #         atom = (
        #             get_energy(filename, num=num)
        #             if num != None
        #             else get_energy(filename)
        #         )  # 可以指定要读取OSZICAR中的第几个离子步的能量
        #     elif "DOSCAR" in car:
        #         atom = get_dos_data(folder_path)
        #     else:
        #         atom = read(filename, format="vasp")
        all_list.append(atom)
        all_dict[subfolder] = atom
    if all(key.isdigit() for key in all_dict):  # 如果键的值都是数字，依据数字顺序调整列表和字典
        all_dict1 = collections.OrderedDict(
            sorted(all_dict.items(), key=lambda item: int(item[0]))
        )
        all_dict = dict(all_dict1)
        all_list = list(all_dict1.values())
    print('1st' ,all_dict.keys())
    return (all_list, all_dict)

def read_cars(source_folder, car="CONTCAR"):
    atoms_l = []
    atoms_d = {}
    atoms_d0 = {}  # 中间变量
    # 获取源文件夹的所有子文件夹
    subfolders = [
        f
        for f in os.listdir(source_folder)
        if os.path.isdir(os.path.join(source_folder, f))
    ]
    for subfolder in subfolders:
        
        subfolder_path = os.path.join(source_folder, subfolder)
        atom_l, atom_d = read_car(subfolder_path, car=car)
        atoms_l.extend(atom_l)
        atoms_d[subfolder] = atom_d  # 字典存字典
        atoms_d0[subfolder] = atom_l  # 字典存列表，用于排序列表
    if all(key.isdigit() for key in atoms_d):  # 如果键的值都是数字，依据数字顺序调整列表和字典
        atoms_d1 = collections.OrderedDict(
            sorted(atoms_d.items(), key=lambda item: int(item[0]))
        )
        atoms_d = dict(atoms_d1)  # 字典重排序

        atoms_d0 = collections.OrderedDict(
            sorted(atoms_d0.items(), key=lambda item: int(item[0]))
        )
        atoms_l = [
            value
            for key, value in sorted(
                atoms_d0.items(), key=lambda item: int(item[0])
            )
        ]
        # 此处的key有意义，保证了value不包含键值
        # 列表重排
    print("2nd", atoms_d.keys())
    return (atoms_l, atoms_d)


# 处理POSCAR函数模块
def read_message(filename):
    # 读取poscar
    message = []
    with open(filename, "r") as f:
        message.extend(line.strip() for line in f)
    return message

def deal_message(message):
    # 处理读取的文本
    mbox, _ = strs(message[2:5])
    mnum, _ = strs(message[6])
    mnum = [[int(element) for element in row] for row in mnum]
    num, info = strs(message[9 : sum(mnum[0]) + 9])
    mdata = num[:, 0:3]
    return mbox, mnum, mdata, info

def strs(mess):
    # deal_message 的子函数
    line = len(mess) if isinstance(mess, list) else 1
    num = []
    info = []
    for i in range(line):
        if isinstance(mess, list):
            number = re.split("[ ,]+", mess[i])
        else:
            number = re.split("[ ,]+", mess)
        if len(number) == 6 and isinstance(number[5], str):
            info.append(number[3:6])
            for m in range(3):
                number[m] = float(number[m])
        else:
            for m in range(len(number)):
                number[m] = float(number[m])
        num.append(number)
    num = np.array(num)
    return num, info

def write_message(message, filename="POSCAR"):
    # 写入poscar
    with open(filename, "w") as f:
        for line in message:
            f.write(line + "\n")
    print(f"write ------ {filename}")

def joint(data, mbox, mnum, mdata, info):
    # 将各部分组合
    newdata = data
    mbox = tostr(mbox)
    newdata[2:5] = mbox

    mnum = tostr(mnum)
    newdata[6] = mnum[0]

    data_info = np.concatenate((mdata, info), axis=1)
    data_info = tostr(data_info)
    newdata[9:] = data_info
    return newdata

def tostr(number_matrix):
    string_matrix = []
    for i in range(len(number_matrix)):
        number = number_matrix[i]
        newline = []
        for m in range(len(number)):
            if len(number) == 2:
                newline.append(str(int(number[m])))
            else:
                newline.append(str(number[m]))

        line = " ".join(newline)
        string_matrix.append(line)
    return string_matrix