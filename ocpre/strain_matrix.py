import random
import pickle

import numpy as np
import matplotlib.pyplot as plt

def poisson_disc_3d(box, num, isplot=False):
    import poisson_disc as pdis
    from mpl_toolkits.mplot3d import Axes3D

    box = np.array(box)
    dims3d = box[:, 1] - box[:, 0]
    cut = ((dims3d[0] * dims3d[1] * dims3d[2]) / 4 / num) ** (1 / 3)
    points3d = pdis.Bridson_sampling(dims3d, radius=cut)
    sort = list(range(len(points3d)))
    random.shuffle(sort)
    sort = sort[:num]
    points3d = points3d[sort]
    points3d = points3d + box[:, 0]
    if isplot == True:
        fig3d = plt.figure()
        # ax3d = Axes3D(fig3d)
        ax3d = Axes3D(fig3d, auto_add_to_figure=False)
        fig3d.add_axes(ax3d)
        ax3d.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2])
        plt.show()
    return points3d

def generate_strain_matrix(
    num_strain,
    all_model_rotate,
    strain_box,
    random_select=0,
    uniform_strain=False,
    inherit=False, # 决定是否从之前保存的文件启动
    dict_name="strain_matrix_dict.pkl",
    array_name="strain_matrix.npy",
):
    if random_select == 0:
        random_select = num_strain

    if inherit:  # load
        try:
            with open(dict_name, "rb") as file:
                strain_matrix_dict = pickle.load(file)
            strain_matrix = np.load(array_name)
            return strain_matrix, strain_matrix_dict
        except FileNotFoundError:
            print("File not found. Generating new strain matrix.")

    if uniform_strain:
        strain = poisson_disc_3d(
            strain_box, num_strain, isplot=False
        )  # generate points
        strain_matrix_dict = {-1: strain}
    else:
        strain = poisson_disc_3d(
            strain_box, all_model_rotate * random_select, isplot=False
        )

    strain_matrix_dict = {}
    strain_matrix = np.zeros((all_model_rotate, random_select, 3))

    for i in range(all_model_rotate):
        if uniform_strain:
            random_numbers = random.sample(range(num_strain), random_select)
            random_numbers.sort()
        else:
            random_numbers = np.array(
                range(i * random_select, (i + 1) * random_select)
            )
        for j in range(len(random_numbers)):
            key = f"{str(i)}_{str(j)}"
            strain_matrix_dict[key] = strain[random_numbers[j]]
        strain_matrix[i] = strain[random_numbers]

    with open(dict_name, "wb") as file:  # save
        pickle.dump(strain_matrix_dict, file)
    np.save(array_name, strain_matrix)

    return strain_matrix, strain_matrix_dict


def test_dict_plot(data_dict):
    # 提取坐标数据
    if isinstance(data_dict, dict):
        print("Input is dict")
        data = list(data_dict.values())
        if isinstance(data[0], dict):
            data_dict = np.concatenate(data, axis=1)
        x_coords = [data_dict[key][0] for key in data_dict]
        y_coords = [data_dict[key][1] for key in data_dict]
        z_coords = [data_dict[key][2] for key in data_dict]
    elif isinstance(data_dict, np.ndarray):
        print("Input is array")
        data_dict = data_dict.reshape(-1,3)
        x_coords = list(data_dict[:, 0])
        y_coords = list(data_dict[:, 1])
        z_coords = list(data_dict[:, 2])
    else:
        raise ValueError("Input not supported")

    # 创建画布和3D轴
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111, projection="3d")
    # 给每个数据点的坐标计数
    coord_count = {}
    for i in range(len(x_coords)):
        coord = (x_coords[i], y_coords[i], z_coords[i])
        if coord in coord_count:
            coord_count[coord] += 5
        else:
            coord_count[coord] = 1
    # 绘制散点图，并根据坐标点数量设置点的大小
    for i in range(len(x_coords)):
        coord = (x_coords[i], y_coords[i], z_coords[i])
        size = 20 + coord_count[coord]  # 设置点的大小
        ax.scatter(
            x_coords[i], y_coords[i], z_coords[i], c="r", marker="o", s=size
        )
    # 设置坐标轴标签
    ax.set_xlabel("strain ε_x", fontsize=12)
    ax.set_ylabel("strain ε_y", fontsize=12)
    ax.set_zlabel("strain 2ε_xy", fontsize=12)  # 显示图形
    plt.show()