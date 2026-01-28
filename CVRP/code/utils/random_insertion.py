import numpy as np
from typing import Optional, Tuple
import numpy.typing as npt

# 类型别名
FloatPointArray = npt.NDArray[np.float32]
IntegerArray = npt.NDArray[np.uint32]
UInt32Array = npt.NDArray[np.uint32]


def tsp_random_insertion_parallel(
        cities: FloatPointArray,
        order: Optional[IntegerArray] = None,
        threads: int = 0
) -> UInt32Array:
    """并行随机插入法求解 TSP（纯 Python 实现）"""
    args, out = _tsp_get_parameters(cities, order, batched=True)
    _random_insertion_impl(*args, out)
    return out


def _random_insertion_impl(
        cities: FloatPointArray,
        order: IntegerArray,
        euclidean: bool,
        out: UInt32Array
) -> None:
    """随机插入法核心实现"""
    batch_size, city_count = order.shape[0], cities.shape[1]

    for b in range(batch_size):
        current_order = order[b].copy() if batch_size > 1 else order.copy()
        remaining = list(current_order[1:])  # 初始路径：[0], 剩余城市：[1, 2, ..., n-1]
        np.random.shuffle(remaining)  # 随机化插入顺序

        path = [current_order[0]]  # 初始路径只有起始城市

        for city in remaining:
            best_pos = 0
            min_increase = float('inf')

            # 遍历所有可能插入位置，找到最小代价增加的位置
            for i in range(len(path) + 1):
                # 计算插入后的新距离
                if i == 0:
                    prev, next_ = path[-1], path[0]
                else:
                    prev, next_ = path[i - 1], path[i % len(path)]

                # 计算距离（欧氏距离或矩阵）
                if euclidean:
                    cost_prev = np.linalg.norm(cities[b, prev] - cities[b, city]) if euclidean else cities[
                        b, prev, city]
                    cost_next = np.linalg.norm(cities[b, city] - cities[b, next_]) if euclidean else cities[
                        b, city, next_]
                    cost_original = np.linalg.norm(cities[b, prev] - cities[b, next_]) if euclidean else cities[
                        b, prev, next_]
                else:
                    cost_prev = cities[b, prev, city]
                    cost_next = cities[b, city, next_]
                    cost_original = cities[b, prev, next_]

                increase = cost_prev + cost_next - cost_original

                if increase < min_increase:
                    min_increase = increase
                    best_pos = i

            path.insert(best_pos, city)  # 插入最佳位置

        out[b] = np.array(path, dtype=np.uint32)


def _tsp_get_parameters(
        cities: FloatPointArray,
        order: Optional[IntegerArray] = None,
        batched: bool = False,
        euclidean: bool = True
) -> Tuple[Tuple[FloatPointArray, IntegerArray, bool], UInt32Array]:
    """参数预处理（与原代码一致）"""
    if batched:
        assert len(cities.shape) == 3
        batch_size, citycount, n = cities.shape
        out = np.zeros((batch_size, citycount), dtype=np.uint32)
    else:
        assert len(cities.shape) == 2
        citycount, n = cities.shape
        out = np.zeros(citycount, dtype=np.uint32)
        cities = cities[np.newaxis, :, :]  # 单批次 -> 多批次

    assert (n == 2) if euclidean else (n == citycount)

    if order is None:
        order = np.arange(citycount, dtype=np.uint32)
        if batched:
            order = np.tile(order, (batch_size, 1))
    else:
        if batched and len(order.shape) == 2:
            assert tuple(order.shape) == (batch_size, citycount)
        else:
            assert len(order.shape) == 1 and order.shape[0] == citycount
            order = np.tile(order, (batch_size, 1)) if batched else order

    _order = np.ascontiguousarray(order, dtype=np.uint32)
    _cities = np.ascontiguousarray(cities, dtype=np.float32)
    return (_cities, _order, euclidean), out