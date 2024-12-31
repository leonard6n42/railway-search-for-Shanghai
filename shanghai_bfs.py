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




# 搜索路径（广度优先搜索）
def bfs_search(adjacency_list, from_station, to_station):
    visited = set()
    queue = [[from_station]]

    while queue:
        path = queue.pop(0)
        node = path[-1]

        if node == to_station:
            return path

        elif node not in visited:
            print(f"Visiting {node}")  # 调试信息
            neighbours = adjacency_list[node]

            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)

                if neighbour == to_station:
                    return new_path

            visited.add(node)

    print(f"No path found between {from_station} and {to_station}.")  # 调试信息
    return None


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
        path = bfs_search(adjacency_list, from_station, to_station)
        if path:
            print(f"Path from {from_station} to {to_station}: {path}")
            # 在绘图时传递路径
            draw_subway_map(adjacency_list, stations_info, path=path)
        else:
            print(f"No path found between {from_station} and {to_station}.")
            draw_subway_map(adjacency_list, stations_info)  # 如果没有找到路径，不传递路径参数

