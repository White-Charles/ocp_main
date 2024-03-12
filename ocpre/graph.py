import torch
import copy
import numpy as np
import pandas as pd
import scipy.spatial.distance as sp_distance

from ase.atoms import Atoms
from spektral.data import Graph
# from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets import LmdbDataset
from ocpre import sort_z

a2g = AtomsToGraphs(
    max_neigh=50,  # 设置每个原子最多与多少个相邻的原子建立联系。
    radius=6,  # 设置建立相邻原子之间边的最大距离。在此半径内的原子将被视为相邻原子。
    r_energy=True,  # False for test data 设置是否将能量作为节点特征添加到图中，默认为True，表示将能量添加到图中。
    r_forces=True,  # 设置是否将力作为节点特征添加到图中，默认为True，表示将力添加到图中。
    r_distances=False,  # 设置是否将原子之间的距离作为特征添加到图中，默认为False，表示不添加距离特征。
    r_fixed=True,  # 设置是否将固定性作为节点特征添加到图中，默认为True，表示将固定性添加到图中
    )
    
def read_trajectory_extract_features(traj):
    if isinstance(traj, list):
        tags = traj[0].get_tags()
        images = [traj[0], traj[-1]]
    elif isinstance(traj, Atoms):
        tags = traj.get_tags()
        images = [traj, traj]
    else:
        raise ValueError("traj must be list or Atoms")
    data_objects = a2g.convert_all(images, disable_tqdm=True)
    data_objects[0].tags = torch.LongTensor(tags)
    data_objects[1].tags = torch.LongTensor(tags)
    return data_objects



def set_energy(model0, energy):
    model = model0.copy()
    model.set_calculator(LennardJones())
    # 假设已知能量为 energy_value
    energy_value = energy  # 假设能量值为10
    # 直接将能量赋值给 atoms 变量
    model.get_potential_energy()  # 先假装计算一次能量
    model.calc.results["energy"] = energy_value  # 修改势能值
    # 输出 atoms 的能量
    # print(model.get_potential_energy())
    return model

def set_tags4adslab(adslab0):
    adslab = adslab0.copy()
    if not isinstance(adslab, Atoms):
        raise TypeError("Variable is not of atomic type")
    tags = np.zeros(len(adslab))
    layers = sort_z(adslab.get_positions()[:, 2])
    filtered_layers0 = [i for i in range(len(layers)) if layers[i] == 6]  # 被吸附基底的表面层
    tags[filtered_layers0] = 1  # 被吸附基底的表面层

    filtered_layers1 = [i for i in range(len(layers)) if layers[i] > 6]  # 被吸附基底的表面层
    tags[filtered_layers1] = 2  # 被吸附基底的表面层
    adslab.set_tags(tags)  # 将tags属性替换
    return adslab

def dict_values_to0(d):
    def replace_values_with_zero(dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                replace_values_with_zero(value)  # 递归调用，处理嵌套字典
            else:
                dictionary[key] = .0  # 将值替换为0    
    d1 = copy.deepcopy(d)
    replace_values_with_zero(d1)
    return(d1)


def get_GNN_data(
    ads_int, int_eng=None, ads_opt=None, opt_eng=None, slab_eng=None, E0_O2=-0.0
):
    """
    # Parameters
    ads_int: initial adslabs without structural optimization
    int_eng: energy of initial adslabs
    ads_opt: structural optimized adslabs
    opt_eng: energy of optimized adslabs
    slab_eng: energy of slabs
    E0_O2: energy of oxygen molecule
    # Returns
    GNN_data_d: a dictionary of adsorb and adsorption energy for GNN ML.
    """
    # 如果是为了预测，那么只有ads_int参数是必须的
    if ads_opt is None:
        ads_opt = copy.deepcopy(ads_int)  # 嵌套字典的复制方法
    if int_eng is None:  # 提取键值，设置能量默认为0
        int_eng = copy.deepcopy(ads_int)
        int_eng = dict_values_to0(int_eng)
    if opt_eng is None:
        opt_eng = copy.deepcopy(ads_int)
        opt_eng = dict_values_to0(opt_eng)
    if slab_eng is None:
        slab_eng = copy.deepcopy(ads_int)
        slab_eng = dict_values_to0(slab_eng)
    GNN_data_d = {}  # 字典
    for key, value in ads_opt.items():
        GNN_data_i = []  # 嵌套列表
        for key0, value0 in value.items():
            # system1
            system_ads_rel = set_tags4adslab(value0)  # 松弛后的构型
            if not isinstance(system_ads_rel, Atoms):
                raise TypeError("Variable is not of atomic type")
            rel_e = opt_eng[key][key0]  # 对应的能量
            ss_e = slab_eng[key][key0]
            # if abs(rel_e - sse) > 0.5 * abs(rel_e):  # 相差太大
            #     sse = slab_eng[key][key0] * 4 * 4
            ads_e = (
                rel_e - ss_e - E0_O2
                if abs(rel_e - ss_e - E0_O2) < abs(rel_e - ss_e * 4 - E0_O2)
                else rel_e - ss_e * 4 - E0_O2
            )
            # 吸附能
            system1 = set_energy(system_ads_rel, ads_e)

            # system0
            system_ads_int = set_tags4adslab(ads_int[key][key0])  # 松弛前的构型
            if not isinstance(system_ads_rel, Atoms):
                raise TypeError("Variable is not of atomic type")
            int_e = int_eng[key][key0]  # 对应的能量
            int_e = int_e - ss_e - E0_O2  # 吸附能
            system0 = set_energy(system_ads_int, int_e)

            data_objects = read_trajectory_extract_features([system0, system1])  #
            initial_struc = data_objects[0]
            relaxed_struc = data_objects[-1]

            initial_struc.y_init = (
                initial_struc.y
            )  # subtract off reference energy, if applicable
            del initial_struc.y
            initial_struc.y_relaxed = (
                relaxed_struc.y
            )  # subtract off reference energy, if applicable
            initial_struc.pos_relaxed = relaxed_struc.pos

            # no neighbor edge case check
            if initial_struc.edge_index.shape == 0:
                print("no neighbors")
                continue

            GNN_data_i.append(initial_struc)

        GNN_data_d[key] = GNN_data_i

    return GNN_data_d

def get_predata (lmdb_path, pre_path):
    '''
    根据地址返回预测值和原本的值
    '''
    dataset= LmdbDataset({"src": lmdb_path})
    energies = torch.tensor([data.y_relaxed for data in dataset])
    sid = torch.tensor([data.sid for data in dataset])
    sorted_index = sorted(range(len(sid)), key=lambda i: sid[i])
    energies=energies[sorted_index]
    
    pre_data = np.load(pre_path)
    pre = pre_data['energy']
    ids = pre_data['ids']
    ids = np.array(list(map(int, ids)))
    sorted_index = sorted(range(len(ids)), key=lambda i: ids[i])
    pre=pre[sorted_index]
    return (energies,pre)


def point_position(line_point1, line_point2, point3):
    # 计算
    # print(line_point1, line_point2, point3)
    if line_point1[0] == line_point2[0]:
        # 处理直线垂直于x轴的情况
        return point3[0] != line_point1[0]
    else:
        # 计算直线的斜率
        slope = (line_point2[1] - line_point1[1]) / (line_point2[0] - line_point1[0])

        # 计算第三个点的纵坐标
        y3 = line_point1[1] + slope * (point3[0] - line_point1[0])

        return point3[1] > y3 
    # return point3[1] > y3+1E-3 # 如果有问题需要考虑是否需要更换
    
def atom2graph(atom, dos, threshold=2.8):
    import scipy.sparse as sp

    pos = atom.pos
    cell = atom.cell
    anum = np.array([int(n) for n in atom.atomic_numbers])

    tri = [
        point_position(cell[0][0], cell[0][1], posi[:2])
        and posi[2] > min(sorted(pos[:, 2] - 0.01, reverse=True)[:18])
        for posi in pos
    ]
    coordinates = np.array(pos[tri])
    anum = np.array(anum[tri])

    # 计算原子之间的距离矩阵
    distances = sp_distance.squareform(sp_distance.pdist(coordinates))
    # 创建连接矩阵
    adjacency = np.zeros((len(coordinates), len(coordinates)), dtype=np.float32)

    if type(threshold) is pd.DataFrame:
        for i in range(len(adjacency)):
            for j in range(len(adjacency)):
                cut = threshold[anum[i]][anum[j]]
                adjacency[i, j] = 1 if distances[i, j] < cut else 0  # 标记成键
    else:
        adjacency[distances <= threshold] = 1

    np.fill_diagonal(adjacency, 0)

    # 创建节点特征矩阵（可选)
    dos1 = dos.reshape((dos.shape[1], -1))
    dos0 = np.zeros((1, dos1.shape[1]))
    dos8 = np.zeros((len(anum), dos1.shape[1]))
    dos8[anum == 8] = dos0
    dos8[anum == 29] = dos1
    node_features = dos8

    # 创建边特征矩阵（可选)
    adj = adjacency.copy()  # 边属性矩阵，来自于链接矩阵
    for i in range(len(anum)):
        if anum[i] == 8:
            for j in range(len(anum)):
                if anum[j] == 8:
                    adj[i, j] = 0  # 不考虑O-O键
    for i in range(len(anum)):
        if anum[i] == 8:
            adj[i, i] = sum(adj[i, :]) + 1  # 计算氧成键
    for i in range(len(anum)):
        if anum[i] != 8:
            for j in range(len(anum)):
                if anum[j] == 8 and adj[i, j] == 1:
                    adj[i, j] = adj[j, j]  # 计算氧成键，1已经分配
    for i in range(len(anum)):
        for j in range(i + 1, len(anum)):
            adj[j][i] = adj[i][j]

    n_edge = int(sum(sum(adjacency)))  # 注意 使用的链接矩阵
    n_edge_features = 4  # 假设每条边有3个属性值
    edges = np.argwhere(adjacency)  # edges包含了所有非零边的坐标
    edge_features = np.zeros((n_edge, n_edge_features))  # 创建一个空的边特征矩阵
    for i, (r, c) in enumerate(edges):
        att = int(max(adjacency[r, c], adj[r, c])) - 1  # 链接矩阵和边属性矩阵的最大值
        if att > 3:
            print(att)
            att = 3
        edge_features[i][att] = 1  # 注意要将属性值从1开始编号转换为从0开始编号

    # 创建标签（可选）
    labels = atom.y_relaxed

    adj_sparse = sp.coo_matrix(adjacency)

    # 创建Graph对象
    graph = Graph(x=node_features, a=adj_sparse, e=edge_features, y=labels)
    return graph