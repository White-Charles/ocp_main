
import itertools
import numpy as np
from ocpre import point_position, sort_z, fix_layers
from ase.data import chemical_symbols

def get_alloy_slab(adslab0, reatoms_str, ):
    pos = adslab0.get_positions()
    cell = adslab0.get_cell()
    # anum = np.array([int(n) for n in adslab0.get_atomic_numbers()])
    an = adslab0.get_atomic_numbers()
    tri = [point_position(cell[0], cell[1], posi[:2])for posi in pos]
    tri = [tri[i] and sort_z(pos[:,2])[i]==5 for i in range(len(tri))]

    # 要将Cu 29 替换成Cr 24，Co 27， Pt 78

    reatoms = [chemical_symbols.index(a) for a in reatoms_str]
    # reatoms_str = ['Cr','Co', 'Pt']
    reatoms_for_cycle = reatoms.copy()
    reatoms_for_cycle.extend(reatoms)
    alloy_slab_l = []
    alloy_slab_d = {}
    an2 = an[tri].copy()
    an2 = np.concatenate((an2,an2)) # .reshape((2,-1))
    for i in range(3): # 替换原子的个数 决定了抓取跨度
        combinations = list(itertools.product(reatoms, repeat=i+1)) # 组合
        combinations = [list(c) for c in combinations]
        cstr = list(itertools.product(reatoms_str, repeat=i+1)) # 组合
        cstr = [list(c) for c in cstr]
        for j in range(len(combinations)): # 替换原子的种类，决定了抓取的起点
            rean = combinations[j]
            for k in range(3): # 旋转 决定了替换的起点，用reshape实现
                if i<2 or k<1:
                    intan = an.copy()
                    an3 = an2.copy()
                    an3[k:k+len(rean)] = rean # 递推
                    an3 = an3.reshape((2,-1))
                    an3[0] = an3[0]-29 # 巧妙地减去了Cu的序号
                    an3 = np.sum(an3, axis=0)
                    intan[tri] = an3
                    test0 = adslab0.copy()
                    test0.set_atomic_numbers(intan)
                    con = {(-1,-1,-1):[-1,-2]} # 约束，除了表面和次表面其他层都约束
                    test0 = fix_layers(test0, fixed_atoms_layers=con)
                    alloy_slab_l.append(test0)
                    
                    asd = '_'.join(map(str, [i,j,k]))
                    alloy_slab_d[asd] = test0
    return alloy_slab_d