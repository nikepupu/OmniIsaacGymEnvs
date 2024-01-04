import os
import json
import numpy as np
from dataclasses import dataclass

@dataclass
class objectInfo:
    usd_path:  str
    position:  np.ndarray
    orientation: np.ndarray
    scale: np.ndarray


def parse_folder(folder_path):
    contact_graph = 'contact_graph_cad.json'
    contact_graph = os.path.join(folder_path, contact_graph)
    with open(contact_graph) as f:
        contact_graph = json.load(f)

    nodes = contact_graph['nodes']
    articulated_nodes = {}
    cad_ids = []
    for node in nodes:
        if 'articulated' in node and node['articulated']:
            articulated_nodes[node['cad_id']] = node
            cad_ids.append(node['cad_id'])
    return articulated_nodes, cad_ids

def cad_id_to_usd_id(folder_path, cad_ids):
    t = {}

    for cad_id in cad_ids:
        path = os.path.join(folder_path, 'articulated', cad_id)
        # find file ends with xacro in path
        suffix = None
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.xacro'):
                    # get file suffix without .xacro
                    tmp = file[:-6]
                    # make sure it's a number
                    if tmp.isnumeric():
                        suffix = tmp
                        t[cad_id] = suffix
                        break
                    

        if suffix is None:
            print('No xacro file found for cad_id: ', cad_id)
            continue

    return t


def usd_paths(folder_path, usd_ids, articulated_nodes):
    t = []
    for name, usd_id in usd_ids.items():
        usd_path = os.path.join(folder_path, usd_id, 'mobility_relabel_gapartnet.usd')
        orientation = articulated_nodes[name]['orientation']
        position = articulated_nodes[name]['position']
        scale = articulated_nodes[name]['scale']

        info = objectInfo(usd_path, position, orientation, scale)
        t.append(info)
    return t
    

if __name__ == '__main__':
    articulated_nodes,cad_ids = parse_folder('/home/nikepupu/Desktop/physcene_1_rigid')
    
    usd_ids = cad_id_to_usd_id('/home/nikepupu/Desktop/physcene_1_rigid', cad_ids)

    objects_to_add = usd_paths('/home/nikepupu/Desktop/physcene_1_rigid', usd_ids, articulated_nodes)
    print(objects_to_add)





