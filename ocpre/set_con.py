import numpy as np
from ase.atoms import Atoms
from ocpre import sort_z
from ase.constraints import FixAtoms, FixCartesian

def set_cons0(model):
    """
    此约束的效果是：固定表面两层以下的原子
    """
    slab = model.copy()
    layers = sort_z(slab.get_positions()[:, 2])
    tags = np.zeros(len(slab))

    filtered_layers2 = [i for i in range(len(layers)) if layers[i] > max(layers) - 2]  #
    tags[filtered_layers2] = 1  # 被吸附基底的次表面和表面

    filtered_layers1 = [i for i in range(len(layers)) if layers[i] > max(layers) - 1]  #
    tags[filtered_layers1] = 2  # 被吸附基底的表面层

    slab.set_tags(tags)  # 将tags属性替换
    # Fixed atoms are prevented from moving during a structure relaxation. We fix all slab atoms beneath the surface
    cons0 = FixAtoms(indices=[atom.index for atom in slab if (atom.tag < 1)])
    cons1 = FixCartesian(filtered_layers2, mask=(1, 1, 0))
    slab.set_constraint([cons0, cons1])
    slab.set_pbc(True)
    return slab


def set_cons1(model):
    """
    此约束的效果是：固定表面一层以下 和 左下角 的原子
    """
    f_pos = model.get_scaled_positions()
    slab = model.copy()
    layers = sort_z(slab.get_positions()[:, 2])
    tags = np.zeros(len(slab))

    filtered_layers2 = [i for i in range(len(layers)) if layers[i] > max(layers) - 2]  #
    tags[filtered_layers2] = 1  # 被吸附基底的次表面和表面

    filtered_layers1 = [i for i in range(len(layers)) if layers[i] > max(layers) - 1]  #
    tags[filtered_layers1] = 2  # 被吸附基底的表面层

    filtered_layers0 = [
        i for i in range(len(model)) if tags[i] == 2 and f_pos[i, 0] + f_pos[i, 1] < 0.5
    ]  #
    tags[filtered_layers0] = -1  # 左下角的三个原子

    slab.set_tags(tags)  # 将tags属性替换
    # Fixed atoms are prevented from moving during a structure relaxation. We fix all slab atoms beneath the surface
    cons0 = FixAtoms(indices=[atom.index for atom in slab if (atom.tag < 2)])
    # cons1 = FixCartesian(filtered_layers2,mask=(1, 1, 0))
    slab.set_constraint([cons0])
    slab.set_pbc(True)
    return slab

def set_cons2(model):
    """
    此约束的效果是：固定所有Cu原子，而氧原子固定x和y方向
    """

    slab = model.copy()
    # Fixed atoms are prevented from moving during a structure relaxation. We fix all slab atoms beneath the surface
    atomic_numbers = slab.get_atomic_numbers()
    cons0 = FixAtoms(
        indices=[slab[i].index for i in range(len(slab)) if atomic_numbers[i] != 8]
    )
    filtered_layers1 = [
        i for i in range(len(atomic_numbers)) if atomic_numbers[i] == 8
    ]  #
    cons1 = FixCartesian(filtered_layers1, mask=(1, 1, 0))
    slab.set_constraint([cons0, cons1])
    slab.set_pbc(True)
    return slab

def set_cons3(model):
    """
    此约束的效果是：固定表面一层以下 和 左下角 的原子，而氧原子固定x和y方向
    """
    model1 = model.copy()
    model0 = model.copy()
    atomic_numbers = model0.get_atomic_numbers()
    filtered_layers1 = [
        i for i in range(len(atomic_numbers)) if atomic_numbers[i] == 8
    ]  #
    del model0[filtered_layers1]  # 先暂时删除氧原子

    model0 = set_cons1(model0)
    cons0 = model0.constraints[0]
    cons1 = FixCartesian([filtered_layers1], mask=(1, 1, 0))
    model1.set_constraint([cons0, cons1])  # 在之前约束的基础上又约束了两个

    # Fixed atoms are prevented from moving during a structure relaxation. We fix all slab atoms beneath the surface
    model1.set_pbc(True)
    return model1

def fix_layers(model,fixed_atoms_layers={}):
    """
    # description
    The effect of this constraint is to fix the cartesian of atoms of the specified layer 
    # parameters
    model: ase.atoms, the atomic model need to set constraint
    fixed_atoms_layers: dictionary, the key means which cartesian would be fixed. Negative means reverse selection of subsequent values.
    the value means which layers would be fixed. Negative means count from bottom.
    {(110):[0,1,2]} means the 0,1,2 layers (count from bottom) need to fix x and y cartesian
    {(-1-1-1):[-1,-2]} means the all layers except 0,1, layers (count from top) need to fix x and y cartesian
    """
    slab = model.copy()
    slab.set_constraint() # 清除约束
    zlayers = sort_z(slab.get_positions()[:, 2]) # z坐标分层
    tags = np.zeros(len(slab))
    alayers =  np.array(range( len( set( zlayers) ) ) ) # 层
    con = [] # 存储约束
    filter = np.empty((len(fixed_atoms_layers)+1),object) # 存储用于固定自由度的标记
    for count, (key, value) in enumerate(fixed_atoms_layers.items(), start=1):
        # key 表示固定方式，正负值表示正选还是反选，value表示选择层，正负表示从下还是从上数
        value0 = alayers[value] # 允许正选和反选
        if sum(key)<0: # 反选
            filtered_layer = [i for i in range(len(zlayers)) if zlayers[i] not in value0]
            tags[filtered_layer] = count
        elif sum(key)>0: # 正选
            filtered_layer = [i for i in range(len(zlayers)) if zlayers[i] in value0]
            tags[filtered_layer] = count
            filter[count] = filtered_layer
    slab.set_tags(tags)  # 将tags属性替换
    for count, (key, value) in enumerate(fixed_atoms_layers.items(), start=1):
        if sum(key) == 3 or sum(key) == -3: # 反选
            cons0 = FixAtoms(indices=[atom.index for atom in slab if (atom.tag == count)])
            con.append(cons0)
        else:
            key0 = tuple(abs(x) for x in key)
            print(key0)
            cons0 = FixCartesian(filter[count], mask=key0)
            con.append(cons0)
    slab.set_constraint(con)
    slab.set_pbc(True)
    return slab