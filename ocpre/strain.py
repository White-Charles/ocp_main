import os
import math
import shutil
import numpy as np

from lammps import lammps
from ase.atoms import Atoms
from ase.build import add_vacuum
from ase.calculators.emt import EMT
from ase.optimize import LBFGS, BFGS
from ase.io import write, read, lammpsdata
from ase.constraints import FixAtoms, FixCartesian
from operator import itemgetter

def move_to_mid(slab):
    if isinstance(slab, Atoms):
        atom = slab.copy()
        pos = atom.get_scaled_positions()
        dis = 0.5 - np.average(pos[:,2])
        pos[:,2] += dis
        atom.set_scaled_positions(pos)
        print(atom.get_positions())
        atom.get_scaled_positions()
    else:
        raise TypeError("Model should be Atoms")
    return(atom)

def cal_strain(ini_atoms, pre_atoms, isprint=True):
    # 函数用于计算应变，变形前模型：ini_atoms, 变形后模型：pre_atoms，两个模型的属性是 ase.atoms.Atoms，
    # 如果两个模型不是Atoms(Type Error)，或者不具有应变变换(Value Error)，会提示错误。

    isAtoms = isinstance(ini_atoms, Atoms) + isinstance(pre_atoms, Atoms)
    len_queal = len(ini_atoms.positions) == len(pre_atoms.positions)
    if isAtoms * len_queal == 0:
        print("Two model are Atoms:", isAtoms == 2)
        print("Two models with equal atomic numbers :", len_queal == 1)
        raise TypeError("Model should be Atoms")
    ini_cor = ini_atoms.cell.array
    pre_cor = pre_atoms.cell.array
    # 计算形变梯度张量
    F = np.matmul(pre_cor, np.linalg.inv(ini_cor))
    E = 1 / 2 * (F.T + F) - np.identity(3)
    if isprint:
        print("strain: \n", E)
    return E

def opt_strain(bulk, strain, iscal=True):

    # 在bulk上施加应变
    strain_bulk = bulk.copy()
    strain_real = 0
    # 获取当前的晶格矩阵, 复制初始无应变构型
    cell = np.array(strain_bulk.get_cell())
    nostrain = np.identity(3)  # 单位矩阵
    # 在 x 方向上添加应变
    F = nostrain + np.array(
        [[strain[0], strain[2], 0], [strain[2], strain[1], 0], [0, 0, 0]]
    )
    newcell = np.matmul(F, cell)
    # 将新的晶格矩阵赋值给 原始 Cu 对象
    strain_bulk.set_cell(newcell, scale_atoms=True)
    # scale_Atoms=True must be set to True to ensure that ...
    # ...the atomic coordinates adapt to changes in the lattice matrix
    if iscal:
        strain_real = extracted_from_opt_strain(
            strain_bulk, lammps, cell, bulk
        )
    else:
        strain_real = cal_strain(bulk, strain_bulk)
    return (strain_bulk, strain_real)

def extracted_from_opt_strain(strain_bulk, lammps, cell, bulk):
    # 施加应变后的模型，lammps可读文件
    write("strain.lmp", strain_bulk, format="lammps-data")  # type: ignore
    # 执行lammps优化，固定了x和y的自由度，只放松了z方向的自由度
    infile = "in.strain.in"
    lmp = lammps()
    lmp.file(infile)
    atoms = lammpsdata.read_lammps_data("opt_strain.data", style="atomic")

    new_cell = atoms.get_cell()
    dot_cell = np.dot(cell[0], cell[1])
    dot_new = np.dot(new_cell[0], new_cell[1])
    if dot_cell * dot_new < 0:  # 与基础的基矢量构型不同
        new_cell[1] = new_cell[1] + new_cell[0]

    strain_bulk.set_cell(new_cell, scale_atoms=True)
    return cal_strain(bulk, strain_bulk)

def cal_LBFGC(ini_model, potential=EMT, fmax=1e-6, steps=1e3):
    # 执行动力学过程，默认的势函数是EMT，力收敛判断值1E-6，最大动力学步数1E3
    # 这个优化似乎不能放缩盒子
    ini_model.set_calculator(potential())  # setting the calculated potential
    # 创建 LBFGS 实例
    dyn = LBFGS(ini_model, logfile="None")
    # 进行能量最小化优化计算
    dyn.run(fmax, steps)
    # 输出优化后的结构信息和能量值
    opt_config = dyn.atoms  # initial model
    opt_energy = dyn.atoms.get_potential_energy()
    return (opt_config, opt_energy)

def cal_BFGC(ini_model, potential=EMT, fmax=1e-6, steps=1000):
    # 执行动力学过程，默认的势函数是EMT，力收敛判断值1E-6，最大动力学步数1E3
    # 这个优化似乎不能放缩盒子
    ini_model.set_calculator(potential())  # setting the calculated potential
    # 创建 BFGS 实例
    dyn = BFGS(ini_model)
    # 进行能量最小化优化计算
    dyn.run(fmax, steps)

    # 输出优化后的结构信息和能量值
    opt_config = dyn.atoms  # initial model
    opt_energy = dyn.atoms.get_potential_energy()
    print("Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm")
    return (opt_config, opt_energy)

def set_strain(ini_model, strain=None, is_opt=True): # 该函数保留未使用
    if strain is None:
        strain = [0, 0, 0]
    # strain 表面应变由三个值控制 [ε1 ε2 ε6]
    isAtoms = isinstance(ini_model, Atoms)
    if isAtoms == 0:
        raise TypeError("Model should be Atoms")

    strain_slab = ini_model.copy()
    # 获取当前的晶格矩阵, 复制初始无应变构型
    cell = strain_slab.get_cell()

    strains = np.array([[strain[0], strain[2]], [strain[2], strain[1]]])
    deform = strains + np.identity(2)
    cell[:2, :2] = np.dot(cell[:2, :2], deform)
    # 将新的晶格矩阵赋值给 原始 Cu 对象
    strain_slab.set_cell(cell, scale_atoms=True, apply_constraint=False)
    # scale_Atoms=True must be set to True to ensure that ...
    # ...the atomic coordinates adapt to changes in the lattice matrix

    if is_opt == True:
        opt_strain_slab, opt_strain_energr = cal_LBFGC(strain_slab)
    else:
        opt_strain_slab, opt_strain_energr = strain_slab, False
    # strain = compare_atoms(ini_model,strain_slab)
    return (opt_strain_slab, opt_strain_energr)

def copy_contcar(
    rootdir,
    destdir=None,
    input_file="CONTCAR",
    output_put="POSCAR",
    func=shutil.copy2,
):
    # 该函数起到读取目录，构建对应目录，执行操作生成文件三个功能。
    # rootdir：已有的目录, destdir=None： 默认的生成的新目录, input_file='CONTCAR'：已有的文件,output_put='POSCAR'：生成的文件
    # func：源文件到新文件之间的操作，默认的操作是复制
    if destdir is None:
        destdir = f"{rootdir} _copy"
    if os.path.exists(destdir):
        shutil.rmtree(destdir)
    poscar_files = []
    for dirpath, _, filenames in os.walk(rootdir):
        # 构建对应的目标文件夹路径
        destpath = dirpath.replace(rootdir, destdir)
        if not os.path.exists(destpath):
            os.makedirs(destpath)
        for filename in filenames:
            if filename == input_file:
                # 构建新的文件名
                new_filename = output_put
                # 构建源文件路径和目标文件路径
                src_file = os.path.join(dirpath, filename)
                dest_file = os.path.join(destpath, new_filename)
                # 复制文件到目标路径
                func(src_file, dest_file)
                poscar_files.append(src_file)
                poscar_files = sorted(poscar_files)
    return poscar_files

def sort_z(data, ft=None):
    # 函数的目的是对z坐标排序，判断对应的坐标所在的层数，有一定的容错。
    if ft is None:
        ft = (max(data) - min(data)) / 1e2  # default fault-tolerant
    s = list(range(len(data)))  # 标记位置
    combined_arr = list(zip(data, s))
    sorted_arr = sorted(combined_arr, key=itemgetter(0))
    data = [x[0] for x in sorted_arr]
    s = [x[1] for x in sorted_arr]
    grouped_data = []
    current_group = []
    grouped_data_s = []
    current_group_s = []
    prev_value = None
    for i in range(len(s)):
        si = s[i]
        value = data[i]
        if prev_value is not None and abs(value - prev_value) >= ft:
            grouped_data.append(current_group)
            grouped_data_s.append(current_group_s)
            current_group = []  # 清空 重新记录
            current_group_s = []
        current_group.append(value)
        current_group_s.append(si)
        prev_value = value
    grouped_data.append(current_group)
    grouped_data_s.append(current_group_s)
    # 按照平均值从小到大排序
    grouped_data_s_with_mean = [
        (sum(group2) / len(group2), group1)
        for group1, group2 in zip(grouped_data_s, grouped_data)
    ]
    sorted_groups2 = sorted(grouped_data_s_with_mean, key=lambda x: x[0])
    # 排序后的类别
    sorted_categories2 = [group for _, group in sorted_groups2]

    tag = [0] * len(data)  # 初始的分类
    for i in range(len(sorted_categories2)):
        for j in sorted_categories2[i]:
            tag[j] = i
    return tag

def build_suface(
    bulk, vacuum_height=15.0, cell_z=None, relax_depth=2, iscala=False
):
    # 默认放松的原子层厚度为2
    if cell_z is not None:
        vacuum_height = cell_z - bulk.get_cell()[-1, -1]
        # 如果指定了期望的最终的cell的z方向高度，将采用期望高度
    # 创建真空层，生成表面
    adslab = bulk.copy()  # 无应变bulk构型
    if len(adslab.constraints):
        _extracted_from_build_suface_9(adslab, relax_depth)
    add_vacuum(adslab, vacuum_height)
    adslab.set_pbc(True)
    if iscala:
        adslab, adslab_e = cal_LBFGC(adslab)
    return adslab

def _extracted_from_build_suface_9(adslab, relax_depth):
    tags = np.zeros(len(adslab))
    layers = sort_z(adslab.get_positions()[:, 2])
    filtered_layers = [
        i for i in range(len(layers)) if layers[i] > max(layers) - relax_depth
    ]  #
    tags[filtered_layers] = 2  # 被吸附基底的表面层
    adslab.set_tags(tags)  # 将tags属性替换
    # Fixed atoms are prevented from moving during a structure relaxation. We fix all slab atoms beneath the surface
    cons = FixAtoms(indices=[atom.index for atom in adslab if (atom.tag < 1)])
    adslab.set_constraint(cons)
    
def vac_ext(atom, vacuum_h=0.0, ads_layer=0):
    # 在原本的bulk上补充多层原子，符合构造规律，然后在z轴增加真空层
    # 相对于只允许复制的方式来扩展晶胞z方向，该命令运行可以很方便地指定扩展的层数
    slab = atom.copy()
    slab2 = slab.copy()
    pos = slab.get_positions()
    layers1 = sort_z(pos[:, 2])
    maxl = 2 + math.ceil(ads_layer / (max(layers1) + 1))
    slab2 = slab2.repeat((1, 1, maxl))
    pos2 = slab2.get_positions()
    layers2 = sort_z(pos2[:, 2])

    filtered_layers = [
        i for i in range(len(layers2)) if layers2[i] > ads_layer + max(layers1)
    ]  # 标记高于需求层之上的部分
    del slab2[filtered_layers]  # 大于需求层数的删去

    slab = slab2.copy()
    slab = build_suface(slab, cell_z=vacuum_h, iscala=False)

    layers3 = sort_z(slab.get_positions()[:, 2])
    filtered_layers2 = [
        i for i in range(len(layers3)) if layers3[i] > max(layers3) - 2
    ]  #
    tags = np.zeros(len(slab))
    tags[filtered_layers2] = 2  # 被吸附基底的表面层
    slab.set_tags(tags)  # 将tags属性替换
    # Fixed atoms are prevented from moving during a structure relaxation. We fix all slab atoms beneath the surface
    cons0 = FixAtoms(indices=[atom.index for atom in slab if (atom.tag < 1)])
    cons1 = FixCartesian(filtered_layers2, mask=(1, 1, 0))
    slab.set_constraint([cons0, cons1])
    slab.set_pbc(True)
    # 打印更新后的结构
    return slab

def operate(src_file, dest_file):
    bulk = read(src_file)
    slab = vac_ext(bulk)
    write(dest_file, slab, format="vasp")