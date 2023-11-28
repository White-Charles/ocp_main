import math
import itertools
import matplotlib.pyplot as plt

from ase.visualize.plot import plot_atoms

def is_diagonal_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    return not any(
        i != j and matrix[i][j] != 0
        for i, j in itertools.product(range(rows), range(cols))
    )

def plot_model(model):
    def is_diagonal_matrix(matrix):
        rows = len(matrix)
        cols = len(matrix[0])
        return not any(
            i != j and abs(matrix[i][j]) > 0.1
            for i, j in itertools.product(range(rows), range(cols))
        )

    cell = model.get_cell()
    if is_diagonal_matrix(cell):
        # 绘制model构型的三视图
        fig, axs = plt.subplots(1, 3, dpi=600)
        fig.subplots_adjust(wspace=0.4, hspace=0)
        # 绘制第一个子图（俯视图）
        axs[0].set_aspect("equal")
        plot_atoms(
            model, axs[0], radii=0.9, rotation=("0x,0y,0z")
        )  # the "rotation" value is the  rotation angle of the axis
        axs[0].set_xlim(-1, cell[0, 0] + cell[1, 0] + 2)
        axs[0].set_ylim(-1, cell[1, 1] + 3)
        # axs[0].quiver(0.8, 0, 0.2, 0, color='r')
        axs[0].set_title("Top view", fontsize=10)
        # 绘制第二个子图（侧视图）
        axs[1].set_aspect("equal")
        plot_atoms(model, axs[1], radii=0.9, rotation=("-90x,0y,0z"))
        axs[1].set_xlim(-1, cell[0, 0] + cell[1, 0] + 4)
        axs[1].set_ylim(-1, cell[2, 2] + 3)
        axs[1].set_title("Front view", fontsize=10)
        # 绘制第三个子图（侧视图）
        axs[2].set_aspect("equal")
        plot_atoms(model, axs[2], radii=0.9, rotation=("-90x,90y,0z"))
        axs[2].set_xlim(-1, cell[1, 1] + 3)
        axs[2].set_ylim(-1, cell[2, 2] + 3)
        axs[2].set_title("Side view", fontsize=10)
        plt.show()
    else:
        cell = model.get_cell()
        # 绘制model构型的三视图
        fig, axs = plt.subplots(1, 3, dpi=600)
        fig.subplots_adjust(wspace=0.4, hspace=0)
        # 绘制第一个子图（俯视图）
        axs[0].set_aspect("equal")
        plot_atoms(
            model, axs[0], radii=0.9, rotation=("0x,0y,0z")
        )  # the "rotation" value is the  rotation angle of the axis
        axs[0].set_xlim(-1, cell[0, 0] + cell[1, 0] + 2)
        axs[0].set_ylim(-1, cell[1, 1] + 3)
        # axs[0].quiver(0.8, 0, 0.2, 0, color='r')
        axs[0].set_title("Top view", fontsize=10)
        # 绘制第二个子图（侧视图）
        axs[1].set_aspect("equal")
        plot_atoms(model, axs[1], radii=0.9, rotation=("-90x,0y,0z"))
        axs[1].set_xlim(-1, cell[0, 0] + cell[1, 0] + 4)
        axs[1].set_ylim(-1, cell[2, 2] + 3)
        axs[1].set_title("Front view", fontsize=10)
        # 绘制第个子图（侧视图）
        axs[2].set_aspect("equal")
        plot_atoms(model, axs[2], radii=0.9, rotation=("-90x,90y,0z"))
        axs[2].set_xlim(-5, cell[0, 0] + cell[1, 0])
        axs[2].set_ylim(-1, cell[2, 2] + 3)
        axs[2].set_title("Side view", fontsize=10)
        plt.show()


def plot_top(model_set, column=3):
    num = len(model_set)
    row = math.ceil(num / column)
    fig, axs = plt.subplots(row, column, dpi=600)
    fig.subplots_adjust(wspace=0.4, hspace=0)
    if row == 1 or column == 1:
        for i in range(num):
            model = model_set[i].copy()
            # 绘制第一个子图（俯视图）
            # print(adslab.cell)
            axs[i].set_aspect("equal")
            plot_atoms(
                model, axs[i], radii=0.8, rotation=("0x,0y,0z")
            )  # the "rotation" value is the  rotation angle of the axis
            axs[i].set_xlim(-1, model.cell[0, 0] + model.cell[1, 0] + 2)
            axs[i].set_ylim(-1, model.cell[1, 1] + 2)
    else:
        for i in range(num):
            a = math.floor(i / column)
            b = i % column
            model = model_set[i].copy()
            # 绘制第一个子图（俯视图）
            # print(adslab.cell)
            axs[a, b].set_aspect("equal")
            plot_atoms(
                model, axs[a, b], radii=0.8, rotation=("0x,0y,0z")
            )  # the "rotation" value is the  rotation angle of the axis
            axs[a, b].set_xlim(-1, model.cell[0, 0] + model.cell[1, 0] + 2)
            axs[a, b].set_ylim(-1, model.cell[1, 1] + 2)
    plt.show()