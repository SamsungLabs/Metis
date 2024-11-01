# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
from typing import List

from utils import GPUNode, DeviceType
from utils import parse_hostfile, parse_nodefile


class GPUCluster:
    def __init__(self, hostfile_path: str, clusterfile_path: str):
        self.host_entries = parse_hostfile(hostfile_path)
        self.nodes_info = parse_nodefile(clusterfile_path)

        self.nodes = dict()
        for node_id, node in self.host_entries.items():
            node_ip = node['ip']
            self.nodes[node_id] = GPUNode(device_type=DeviceType.from_string(self.nodes_info[node_ip]['instance_type']),
                                          num_devices=node['num_device'])

    def get_num_nodes(self) -> int:
        return len(self.nodes.keys())

    def get_num_nodes_by_device_type(self, device_type: str) -> int:
        return sum([self.nodes[node_id].num_devices for node_id in self.nodes.keys() if self.nodes[node_id].device_type.name == device_type])

    def get_num_devices_per_node(self) -> int:
        return self.nodes[0].num_devices

    def get_total_num_devices(self) -> int:
        num_devices_list = [self.nodes[node_id].num_devices for node_id in self.nodes.keys()]
        return sum(num_devices_list)

    def get_device_types(self) -> List[DeviceType]:
        return [self.nodes[node_id].device_type for node_id in self.nodes.keys()]

    def get_str_device_types(self) -> str:
        return '_'.join([device_type.name for device_type in set(self.get_device_types())])

    def get_device_memory(self, node_id: int) -> int:
        """
        returns the total memory size of a single GPU within node
        :param node_id:
        :return: Memory size in bytes
        """
        node_ip = self.host_entries[node_id]['ip']
        return self.nodes_info[node_ip]['memory'] * 1024

    def get_device_memory_for_device_type(self, device_type: str) -> int:
        for node_ip in self.nodes_info.keys():
            if self.nodes_info[node_ip]['instance_type'] == device_type:
                return self.nodes_info[node_ip]['memory'] * 1024

    def get_intra_bandwidth(self, node_id: int) -> int:
        node_ip = self.host_entries[node_id]['ip']
        return self.nodes_info[node_ip]['intra_bandwidth']

    def get_inter_bandwidth(self, node_id: int) -> int:
        node_ip = self.host_entries[node_id]['ip']
        return self.nodes_info[node_ip]['intra_bandwidth']
