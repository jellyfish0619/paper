import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional  # 添加 Optional 导入
from matplotlib.figure import Figure


def visualize_heatmap(
        heatmap: Union[np.ndarray, "torch.Tensor"],
        title: str = "Heatmap Visualization",
        figsize: tuple = (20, 16),
        cmap: str = "viridis",
        annot: bool = False,
        cbar_label: str = "Value",
        vmin: Optional[float] = None,  # 现在 Optional 已定义
        vmax: Optional[float] = None,
        dpi: int = 300,
        save_path: Optional[str] = None,
) -> Figure:
    """
    针对大尺寸 heatmap 优化的可视化函数（500+节点）。

    Args:
        heatmap: 输入的 2D 矩阵（支持 numpy 或 torch.Tensor）
        title: 图像标题
        figsize: 图像尺寸（根据节点数调整）
        cmap: 颜色映射（推荐 "viridis", "plasma", "magma"）
        annot: 是否显示数值（大矩阵建议关闭）
        cbar_label: 颜色条标签
        vmin/vmax: 手动设置颜色范围
        dpi: 图像分辨率
        save_path: 保存路径（如 "heatmap.png"）

    Returns:
        matplotlib.figure.Figure
    """
    # 转换为 numpy 数组
    if hasattr(heatmap, "detach"):
        heatmap_np = heatmap.detach().cpu().numpy()
    else:
        heatmap_np = np.array(heatmap)

    # 创建画布
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 绘制热力图
    sns.heatmap(
        heatmap_np,
        ax=ax,
        cmap=cmap,
        annot=annot,
        fmt=".2f",
        cbar_kws={"label": cbar_label},
        vmin=vmin,
        vmax=vmax,
        square=True,
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Node Index", fontsize=12)
    ax.set_ylabel("Node Index", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"Heatmap saved to {save_path}")

    return fig