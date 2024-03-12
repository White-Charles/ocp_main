import os
import re
import shutil
import collections
import numpy as np
import matplotlib.pyplot as plt

from ase.atoms import Atoms
from vaspvis.dos import Dos
from vaspvis import standard
from ase.io import write, read
from ase.constraints import FixAtoms, FixCartesian

def write_atoms(filename, atoms, format="vasp", isp=False):
    # 该函数的作用是修正 ase 的 write 函数，可以输出固定特定原子自由度的约束
    write(filename, atoms, format=format)
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
    write_message(newdata, filename=filename,isp=isp)  # 输出poscar
    write(filename, read(filename), format="vasp")  # 读写检验
    return "write FixCartesian"

def out_poscar(input, path="New_Folder", isp=False):
    # 如果文件夹已经存在，则删除该文件夹及其中的内容
    if os.path.exists(path):
        shutil.rmtree(path)
    # 创建新的文件夹
    os.makedirs(path)

    if isinstance(input, dict):
        if isp:
            print("input is dict")
        all_model = input
    elif isinstance(input, list):
        if isp:
            print("input is list")
        all_model = {str(int(index+1e3)): value for index, value in enumerate(input)}    
    elif isinstance(input, np.ndarray):
        if isp:
            print("input is ndarray")
        all_model = {str(int(index+1e3)): value for index, value in enumerate(input)}
    elif isinstance(input, Atoms):
        if isp:
            print("input is Atoms")
        atoms = input
        filepath = path
        filename = os.path.join(filepath, "POSCAR")
        write_atoms(filename, atoms, format="vasp", isp=isp)
    else:
        raise ValueError("input must be a list or dict or np.ndarray or atoms")
    
    if isinstance(input, Atoms) == False:
        for key in all_model.keys():
            atoms = all_model[key]
            filepath = os.path.join(".", path, key)
            os.makedirs(filepath)                                                                                       
            filename = os.path.join(filepath, "POSCAR")
            write_atoms(filename, atoms, format="vasp",isp=isp)

def out_car_list(input, path="New_Folder", isp=False):
    # 这个函数中嵌套了out_poscar，所以实际输出是嵌套文件夹
    # 如果文件夹已经存在，则删除该文件夹及其中的内容
    if os.path.exists(path):
        shutil.rmtree(path)
    # 创建新的文件夹
    os.makedirs(path)
    
    if isinstance(input, dict):
        if isp:
            print("all input is dict")
        all_model = input
    elif isinstance(input, list):
        if isp:
            print("all input is list")
        all_model = {str(int(index+1e3)): value for index, value in enumerate(input)}    
    elif isinstance(input, np.ndarray):
        if isp:
            print("all input is ndarray")
        all_model = {str(int(index+1e3)): value for index, value in enumerate(input)}
    else:
        raise ValueError("input must be a list or dict or np.ndarray")
    
    for key in all_model.keys():
        filepath = os.path.join(".", path, key)
        os.makedirs(filepath)
        out_poscar(all_model[key], path=filepath, isp=isp)


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

def get_force(OUTCAR_file):
    with open(OUTCAR_file, 'r') as file:
        data = file.read()
    # 找到起始和结束标识符的行号
    start = 'TOTAL-FORCE (eV/Angst)'
    end = 'total drift'
    lines = data.split('\n')
    for i, line in enumerate(lines):
        if start in line:
            start_line = i + 2
        if end in line:
            end_line = i - 1
    # 提取数据部分
    extracted_data = '\n'.join(lines[start_line:end_line]).strip()
    # 将提取的数据转换成 np 数组
    force = np.array([list(map(float, line.split())) for line in extracted_data.split('\n')])
    force = force[:,3:]
    return(force)

def read_one_car(path, car="CONTCAR", isp=False):
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
    # print(path)
    if type(car) == list:
        num = car[1]
        car = car[0]
    else:
        num = -1 
    if callable(car): # 如果car是一个函数
        atom = car(path)
    else:
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
            elif "OUTCAR" in car:
                atom = get_force(filename)
            else:
                atom = read(filename, format="vasp")
    if isp:
        print(filename)
    return (atom)

def read_car(path, car="CONTCAR",isp=False):
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
    
    all_list = []
    all_dict = {}
    subfolders = [
        f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))
    ]
    for subfolder in subfolders:
        folder_path = os.path.join(path, subfolder)
        atom = read_one_car(folder_path, car, isp=isp)
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

def read_cars(source_folder, car="CONTCAR",isp=False):
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
        atom_l, atom_d = read_car(subfolder_path, car=car,isp=isp)
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

def get_dos_data(
    dos_folder, atom_num=[6], orbital_list=list(range(9)), erange=[-8, 8],no_plot=True
):  
    # print(dos_folder)
    # 该函数可以获得指定文件夹的dos信息
    # 默认是POSCAR的第七个原子（atom_num=6），9个轨道，能量范围[-11,11]eV
    atom_orbital_dict = {}
    for i in atom_num:
        atom_orbital_dict[i] = orbital_list
    print(atom_orbital_dict)
    dos_data = standard.dos_atom_orbitals(
        atom_orbital_dict=atom_orbital_dict,
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

def get_dos_atom_orbitals(
    folder,
    atoms,
    erange=[-8.0, 8.0],
    spin="up",
    combination_method="add",
    shift_efermi=0,
):
    """
    This function plots the orbital projected density of states on specific atoms.

    Parameters:
        folder (str): This is the folder that contains the VASP files
        atom_orbital_dict (dict[int:list]): A dictionary that contains the individual atoms and the corresponding
            orbitals to project onto. For example, if the user wants to project onto the s, py, pz, and px orbitals
            of the first atom and the s orbital of the second atom then the dictionary would be {0:[0,1,2,3], 1:[0]}
        combination_method (str): If spin == 'both', this determines if the spin up and spin down
            desnities are added or subtracted. ('add' or 'sub')
    Returns:
        dos(np.array): 
    """
    # 这是一个修改的函数，抛弃了其中绘图的部分，保留了其中提取数据的部分
    
    dos = Dos(
        shift_efermi=shift_efermi,
        folder=folder,
        spin=spin,
        combination_method=combination_method,
    )
    
    dos_data = dos.pdos_array[:,atoms,:]
    c = dos.tdos_array[:,0]
    indices = np.where((c > erange[0]) & (c <= erange[-1]))
    return(c[indices], dos_data[indices])

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
    mnum = [[int(float(element)) for element in row] for row in mnum]
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

def write_message(message, filename="POSCAR", isp=False):
    # 写入poscar
    with open(filename, "w") as f:
        for line in message:
            f.write(line + "\n")
    if isp:
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

def array2dict(arr):
    '''
    可以将一个矩阵，转成一个字典，输入可以是多维字典，那么输出是嵌套矩阵
    '''

    dict_arr = arr.copy()
    if isinstance(dict_arr, np.ndarray):
        while dict_arr.shape != ():
            shape = dict_arr.shape
            dict_arr = [dict(zip([str(i+1000) for i in range(shape[-1])], row)) for row in dict_arr.reshape((-1, shape[-1]))]
            dict_arr = np.array(dict_arr).reshape(shape[:-1])
        dict_data = dict_arr.tolist()
    
    elif isinstance(dict_arr, list):
        dict_data = {}
        for i, item in enumerate(dict_arr):
            dict_data[str(i+1000)] = list_to_nested_dict(item)
            
    return(dict_data)

def list_to_nested_dict(lst):
    if not isinstance(lst, list):
        return lst

    dict_data = {}
    for i, item in enumerate(lst):
        dict_data[str(i+1000)] = list_to_nested_dict(item)
    
    return dict_data

def dict2list(nested_dict):
    if isinstance(nested_dict, dict):
        return [dict2list(value) for value in nested_dict.values()]
    else:
        return nested_dict