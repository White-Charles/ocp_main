import math
import numpy as np
import matplotlib.pyplot as plt

from ase.io import  read
from ase.atoms import Atoms
from ase.build import add_adsorbate, molecule, fcc111
from ase.constraints import FixAtoms, FixCartesian
from ocpre import read_one_car, set_cons2, cal_LBFGC, array2dict, dict2list, set_cons1, opt_strain, vac_ext, write_atoms, point_position, sort_z, fix_layers



def int_bulk(path):
    CONT = read_one_car(path)
    if isinstance(CONT, Atoms):
        CONT_cell = CONT.get_cell()
        LC = np.array(CONT_cell)[0, 0]
    else:
        LC = 3.615  # lattice of Cu
        # LC = 3.48 # lattice of Ni
    print(LC)
    iforthogonal = False  # 是否正交
    model = fcc111("Cu", size=(1, 1, 3), a=LC, orthogonal=iforthogonal, periodic=True)
    # 注意，这里还没有增加真空层，size[2]必须是3的倍数，这样才能保证bulk在z方向的周期性
    model.set_pbc(True)  # 最早的初始构型
    # strain = cal_strain(model, model)
    # cell = model.get_cell()
    bulk = model.copy()
    return bulk


def get_bulk_set(onebulk, strain_matrix):
    '''
    Based on strain, obtain the set of bulk applied with strain load.
    根据应变，获得施加应变的bulk的集合
    
    Parameters
    ----------
    onebulk : Atoms
        Strain-Free Box
    strain_matrix : array
        Input array.

    Returns
    -------
    bulk_matrix : ndarray
        The output array.  The number of dimensions one less than the `strain_matrix`,
        but the shape can be same except for the last one-dimensional. 
    '''
    
    def strain_load(onestrain): # 施加应变
        bulk = opt_strain(onebulk, onestrain, iscal=False)[0]
        return bulk
    def get_Atoms(atom): # 修正，提取起atom的atoms属性
        return atom.atoms
    get_Atoms_vector = np.vectorize(get_Atoms) # 函数向量化
    bulk_matrix = np.apply_along_axis(strain_load, axis=-1, arr=strain_matrix) 
    bulk_matrix = get_Atoms_vector(bulk_matrix[...,0])
    bulk_list = bulk_matrix.tolist()
    bulk_dict = array2dict(bulk_list)
    return bulk_list, bulk_dict


def operate(src_file, dest_file):
    bulk = read(src_file)
    slab = vac_ext(bulk, vacuum_h=30.0, ads_layer=4)
    # slab = move_to_mid(slab)
    write_atoms(dest_file, slab, format="vasp")

    
def get_dic(adslab, atom_model):
    tags = adslab.get_tags()
    position = adslab.positions
    # top = position[tags >= 1]
    frac_pos = adslab.get_scaled_positions()
    # frac_top = frac_pos[tags >= 1]
    tags = adslab.get_tags()  # 原子标签
    mask = (
        np.logical_or(tags == 2, tags == -1)
        & (frac_pos[:, 0] > 0.5)
        & (frac_pos[:, 1] > 0.5)
    )
    four_a = position[mask]
    # four_a = top[(frac_top[:,0]>0.3) & (frac_top[:,0]<0.7)
    #                 & (frac_top[:,1]>0.3)& (frac_top[:,1]<0.7)] # 找到需要的四个原子
    points = four_a[:, :2]  # 4个原子的x和y坐标
    
    from sklearn.neighbors import KDTree

    # 创建一个 KDTree 对象，用于快速搜索最近邻点
    tree = KDTree(points)
    # 使用列表保存每个点的邻居
    neighbor_list = []
    relation_list = []
    for i in range(len(points)):
        # 找到该点距离小于等于 2 的所有邻居
        neighbor = tree.query_radius([points[i]], r=3)[
            0
        ]  # r=3 it is need less then lattice 3.615
        # 排除掉自身，并将结果加入邻居列表中
        # neighbor = [n for n in neighbor if n != i]
        neighbor_list.append(neighbor)
        neighbor = [n for n in neighbor if n > i]
        relation_list.append(neighbor)  # 只统计比自己大的邻居，避免重复
    # 获取相邻元素个数大于 2 的元素
    tri2 = [n for n in neighbor_list if len(n) < 4]
    top_dict = {f"top_{str(i)}": points[i] for i in range(len(points))}
    bri_dict = {}  # bridge 位置的坐标
    for i in range(len(relation_list)):  # i < j 之前定义好了
        for j in relation_list[i]:
            bri_dict[f"bri_{str(i)}{str(j)}"] = np.mean(points[[i, j], :], axis=0)
            xs, ys = zip(*points[[i, j], :])
            # 使用 plot 函数将点连成一条线
            plt.plot(xs, ys, color="red", linewidth=1)
    hol_dict = {}  # hollow 位置的坐标
    for item in tri2:
        tri1 = item
        tri = points[tri1]
        hol_dict[f"hol_{str(tri1[0])}{str(tri1[1])}{str(tri1[2])}"] = np.mean(
            tri, axis=0
        )
    plt.plot(points[:, 0], points[:, 1], "o")
    all_dict = [top_dict, bri_dict, hol_dict]
    for i in range(3):
        data = all_dict[i]
        for key, value in data.items():
            plt.scatter(value[0], value[1], s=100)
            plt.text(value[0], value[1], key, fontsize=15)
    # 显示图形
    plt.axis("equal")
    plt.show()
    # 依据单原子的吸附高度补充字典中z方向的值
    mel_dict = {}
    for i in range(2):
        data = all_dict[i]
        z_value = atom_model[i].positions[-1, -1]
        z_value = np.array([z_value])
        for key, value in data.items():
            data[key] = np.concatenate((data[key][:2], z_value))  # 调整z值
        mel_dict |= all_dict[i]
    # python 的变量具有地址，所以内部的三个字典本体也发生了变化。

    data = all_dict[2]  # 其中是 hol_012(FCC) 和 hol_123(HCP)
    for i, (key, value) in enumerate(data.items(), start=2):
        z_value = atom_model[i].positions[-1, -1]
        z_value = np.array([z_value])
        data[key] = np.concatenate((data[key][:2], z_value))  # 调整z值
    mel_dict |= all_dict[2]
    return mel_dict

def find_nearest_coordinate(aspos, spos):
    # 函数的作用是找到 adsatoms 中所有原子在slabatoms中的近邻原子
    # 将坐标数组转换为 NumPy 数组
    nei = []
    for i in range(len(aspos)):
        apos = aspos[i]
        distances = np.linalg.norm(apos - spos, axis=1)
        mdis = np.min(distances)
        indices = np.where(distances <= mdis + 1e-3)
        nei.append(indices[0])
    return nei

def plot_points(a, b):
    # 第一组点
    x1 = a[:, 0]
    y1 = a[:, 1]

    # 第二组点
    x2 = b[:, 0]
    y2 = b[:, 1]
    # 创建图形对象和子图对象
    fig, ax = plt.subplots()

    # 设置x和y轴等比例
    ax.set_aspect("equal")
    # 设置x轴和y轴范围
    ax.set_xlim([5, 15])
    ax.set_ylim([0, 10])
    # 绘制第一组点
    # 绘制第一组点
    ax.plot(x1, y1, "ro", markersize=15, label="Cu")
    # 绘制第二组点，使用空心圆圈
    ax.plot(x2, y2, "wo", mec="b", mew=2, label="O2")

    # 添加图例
    ax.legend()

    # 显示图形
    plt.show()

def get_atom_adsmodel(adslab,mask,order=None):
    positions = adslab.get_positions()  # 笛卡尔坐标
    top = positions[mask]
    
    # 分别要计算四个单原子的参考位置
    pos = np.zeros((4, 3))
    for i in range(4):
        pos[i] = np.mean(top[max([i - 2, 0]) : i + 1], axis=0)
        # 一个 两个 三个 原子的平均，对应的是吸附原子的位置
    pos[:, 2] = 0
    pos  # 三个原子（top bridge FCC HCP）的x和y的位置
    
    ads = molecule("O", positions=[(0, 0, 0)])
    atom_model_cala = []
    if order is None:
        order = [str(i) for i in range(len(mask))]
    height = 1.0
    for i in range(4):
        adslab0 = adslab.copy()
        position = pos[i, :2]
        offset = (0, 0)
        add_adsorbate(adslab0, ads, height, position=position, offset=offset)

        adslab0 = set_cons2(adslab0)
        adslab0, _ = cal_LBFGC(adslab0)
        atom_model_cala.append(adslab0)
        
        print(adslab0.get_positions()[-1])
        
    atom_model_dict = dict(zip(order, atom_model_cala))
    
    return (atom_model_cala , atom_model_dict)

def get_top3d_rotate(adslab, top3d_index):
    tags = adslab.get_tags()
    index = np.array(range(len(adslab)))
    indextop = index[np.logical_or(tags == 2, tags == -1)]
    
    top3d = [
        [indextop[i] for i in top3d_index[0]],
        [indextop[i] for i in top3d_index[1]],
        [indextop[i] for i in top3d_index[2]],
    ]
    # 对应的是四个原子3个旋转角度的参考原子 重要
    
    return(top3d)

# 下面这两个函数有联系，bulk2slab可以将bulk列表转为slab列表，slab2slabs可以将1*1slab列表转为4*4列表。
def bulk2slab(bulk_list):
    slab_list = [] # 在bulk上施加真空层
    for i in range(len(bulk_list)):
        refer_slab_ml_i = []
        refer_slab_ml_i
        for j in range(len(bulk_list[i])):
            bulk = bulk_list[i][j]
            slab = vac_ext(bulk, vacuum_h=30.0, ads_layer=4)
            refer_slab_ml_i.append(slab)
        slab_list.append(refer_slab_ml_i)
    return(slab_list)

def slab2slabs(slab,repeat_multipy=(4,4,1)):
    strain_list1 = slab.copy()  # 不用copy会改变原本的变量
    slabs = []
    for i in range(len(strain_list1)):
        ssl = []
        for j in range(len( strain_list1[i])):
            value = strain_list1[i][j]
            value_copy = value.copy()
            value_copy.set_constraint()
            value_copy = value_copy.repeat(repeat_multipy)
            ssl.append(value_copy)
        slabs.append(ssl)
    slabs_dict = array2dict(slabs)
    return(slabs, slabs_dict)

def rotate_point(points, axis, angle=120):
    new_points = points
    for i in range(len(points)):
        point = points[i]
        x = point[0]
        y = point[1]
        a = axis[0]
        b = axis[1]
        theta = math.radians(angle)  # 计算旋转角度，将角度转换为弧度
        x_new = a + (x - a) * math.cos(theta) - (y - b) * math.sin(theta)
        y_new = b + (x - a) * math.sin(theta) + (y - b) * math.cos(theta)
        new_point = np.array([x_new, y_new])
        new_points[i] = new_point
    return new_points

def get_molecule_adslab(adslab, mol):
    mol_num = len(mol)
    molecule_model = []
    for i in range(mol_num):
        ads = molecule("O2", positions=mol[i])
        adslab0 = adslab.copy()
        height = mol[i][0][-1] - adslab.positions[-1][-1]
        add_adsorbate(adslab0, ads, height, position=(mol[i][0][:2]), offset=(0, 0))
        adslab0 = set_cons2(adslab0)
        molecule_model.append(adslab0)
        print(adslab0.positions[-2:, -1])
    return(molecule_model)

def get_strain_adslab(mol_rotate_dic, strain_slabs_d2, sort_nei):
    strain_slabs_l2 = dict2list(strain_slabs_d2)
    strain_adslab = []
    for i, (key, mol_refer) in enumerate(mol_rotate_dic.items()):  # 取出来一个无应变的吸附基底
        print(key)
        # strain_slab_0 = strain_slabs_d2[str(i + 1000)]
        strain_adslabk = []
        for l in range(len(strain_slabs_l2[0])):
            sslab_model = strain_slabs_d2[str(i + 1000)][str(l + 1000)]  # 取出来一个应变基底
            sslab_model = set_cons1(sslab_model)
            O2 = mol_refer[mol_refer.get_atomic_numbers() == 8]  # 提取其中的氧原子
            O2_pos = O2.get_positions()
            for j in range(len(O2)):  # 依据参考位置调整氧分子坐标
                pos_s = sslab_model[sort_nei[i][j]].get_positions()
                pos_s = np.mean(pos_s, axis=0)
                pos_m = mol_refer[sort_nei[i][j]].get_positions()
                pos_m = np.mean(pos_m, axis=0)
                pos_d = np.array(pos_s) - np.array(pos_m)
                O2_pos[j] += pos_d

            # 吸附氧分子
            O2 = Atoms("O2")
            O2.set_positions(O2_pos)
            sslab_model.extend(O2)
            # 设置自由度
            cons0 = sslab_model.constraints[0]
            cons1 = FixCartesian(
                [len(sslab_model) - 2, len(sslab_model) - 1], mask=(1, 1, 0)
            )
            sslab_model.set_constraint([cons0, cons1])  # 在之前约束基础上约束了两个
            strain_adslabk.append(sslab_model)
        strain_adslab.append(strain_adslabk)
    return (strain_adslab)