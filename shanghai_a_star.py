from heapq import heappop, heappush
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import matplotlib

# 设置中文字体以确保正确显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取并解析CSV文件中的地铁线路和站点信息
def parse_subway_data_from_csv(file_path):
    df = pd.read_csv(file_path)

    lines_info = {}  # ‘线路’：‘所有站点’
    stations_info = {}  # 站点经纬度信息

    for index, row in df.iterrows():
        station_name = row['站点']
        longitude = float(row['lon'])
        latitude = float(row['lat'])
        line_name = row['所属线路']

        if station_name not in stations_info:
            stations_info[station_name] = [longitude, latitude]

        if line_name not in lines_info:
            lines_info[line_name] = []
        if station_name not in lines_info[line_name]:
            lines_info[line_name].append(station_name)

    # 确保每个线路的站点按顺序排列（如果CSV中没有顺序信息，则需要额外处理）
    for line_name, stations in lines_info.items():
        lines_info[line_name] = sorted(stations, key=lambda x: stations.index(x))

    return lines_info, stations_info


# 构建邻接表
def build_adjacency_list(lines_info):
    adjacency_list = {}

    # 初始化所有站点
    for line, stations in lines_info.items():
        for station in stations:
            if station not in adjacency_list:
                adjacency_list[station] = set()

    # 构建邻接关系
    for line, stations in lines_info.items():
        for i, station in enumerate(stations):
            if i > 0:
                prev_station = stations[i - 1]
                adjacency_list[station].add(prev_station)
                adjacency_list[prev_station].add(station)
            if i < len(stations) - 1:
                next_station = stations[i + 1]
                adjacency_list[station].add(next_station)
                adjacency_list[next_station].add(station)

    return adjacency_list


# 计算两个站点之间的直线距离（Haversine公式）
def haversine_distance(station_info, from_station, to_station):
    R = 6371.0  # 地球半径，单位为公里

    lon1, lat1 = map(radians, station_info[from_station])
    lon2, lat2 = map(radians, station_info[to_station])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# A* 搜索算法
def a_star_search(adjacency_list, stations_info, from_station, to_station):
    open_set = []  # 优先队列，存储待探索的节点
    came_from = {}  # 记录每个节点是从哪个节点到达的
    g_score = {station: float('inf') for station in adjacency_list}  # 到达各节点的最小成本
    f_score = {station: float('inf') for station in adjacency_list}  # 启发式的总估计成本

    g_score[from_station] = 0
    f_score[from_station] = haversine_distance(stations_info, from_station, to_station)

    heappush(open_set, (f_score[from_station], from_station))

    while open_set:
        _, current = heappop(open_set)

        if current == to_station:
            return reconstruct_path(came_from, current)

        for neighbour in adjacency_list[current]:
            tentative_g_score = g_score[current] + haversine_distance(stations_info, current, neighbour)

            if tentative_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g_score
                f_score[neighbour] = tentative_g_score + haversine_distance(stations_info, neighbour, to_station)
                if not any(neighbour == node[1] for node in open_set):
                    heappush(open_set, (f_score[neighbour], neighbour))

    print(f"No path found between {from_station} and {to_station}.")  # 调试信息
    return None


# 重构路径
def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from.keys():
        current = came_from[current]
        total_path.append(current)
    total_path.reverse()
    return total_path


def draw_subway_map(adjacency_list, stations_info, path=None):
    G = nx.Graph(adjacency_list)
    pos = {station: (info[0], info[1]) for station, info in stations_info.items()}

    plt.figure(figsize=(15, 10))

    # 绘制所有站点和连线
    nx.draw(G, pos, with_labels=True, node_size=30, font_size=8, font_weight='bold', node_color='skyblue')

    if path:
        # 提取路径中的边
        path_edges = list(zip(path, path[1:]))

        # 高亮显示路径上的站点和连线
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=60, node_color='red')
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

        # 标记起始站和终点站
        nx.draw_networkx_nodes(G, pos, nodelist=[path[0]], node_size=60, node_color='green', alpha=0.7)
        nx.draw_networkx_nodes(G, pos, nodelist=[path[-1]], node_size=60, node_color='purple', alpha=0.7)

        # 添加标签给起始站和终点站
        nx.draw_networkx_labels(G, pos, labels={path[0]: f"起点\n{path[0]}", path[-1]: f"终点\n{path[-1]}"}, font_size=8, font_weight='bold')

    plt.title('上海地铁图')
    plt.show()


if __name__ == "__main__":
    file_path = '上海市地铁站点.csv'  # CSV文件路径
    lines_info, stations_info = parse_subway_data_from_csv(file_path)
    adjacency_list = build_adjacency_list(lines_info)

    from_station = '周浦东'
    to_station = '剑川路'

    if from_station not in adjacency_list or to_station not in adjacency_list:
        print(f"Either {from_station} or {to_station} does not exist in the adjacency list.")
    else:
        path = a_star_search(adjacency_list, stations_info, from_station, to_station)
        if path:
            print(f"Path from {from_station} to {to_station}: {path}")
            # 在绘图时传递路径
            draw_subway_map(adjacency_list, stations_info, path=path)
        else:
            print(f"No path found between {from_station} and {to_station}.")
            # 如果没有找到路径，不传递路径参数
            draw_subway_map(adjacency_list, stations_info)