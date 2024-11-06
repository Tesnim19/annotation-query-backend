import json
from biocypher import BioCypher
import logging
import yaml
from typing import Dict

# Setup basic logging
logging.basicConfig(level=logging.DEBUG)

class SchemaManager:
    def __init__(self, schema_config_path: str, biocypher_config_path: str):
        self.bcy = BioCypher(schema_config_path=schema_config_path, biocypher_config_path=biocypher_config_path)
        self.schema = self.process_schema(self.bcy._get_ontology_mapping()._extend_schema())
        self.parent_nodes =self.parent_nodes()
        self.parent_edges =self.parent_edges()
        self.graph_info = self.get_graph_info()
        print(self.graph_info)
    
    def process_schema(self, schema):
        process_schema = {}

        for value in schema.values():
            input_label = value.get("input_label")
            output_label = value.get("output_label")
            source = value.get("source")
            target = value.get("target")

            labels = output_label or input_label
            labels = labels if isinstance(labels, list) else [labels]
            sources = source if isinstance(source, list) else [source]
            targets = target if isinstance(target, list) else [target]

            for i_label in labels:
                for s in sources:
                    for t in targets:
                        key_label = f'{s}-{i_label}-{t}' if s and t else i_label
                        process_schema[key_label] = {**value, "key": key_label}

        return process_schema

    
    def parent_nodes(self):
        parent_nodes = set()
        for _, attributes in self.schema.items():
            if 'represented_as' in attributes and attributes['represented_as'] == 'node' \
                    and 'is_a' in attributes and attributes['is_a'] not in parent_nodes:
                parent_nodes.add(attributes['is_a'])
        return list(parent_nodes)

    def parent_edges(self):
        parent_edges = set()
        for _, attributes in self.schema.items():
            if 'represented_as' in attributes and attributes['represented_as'] == 'edge' \
                    and 'is_a' in attributes and attributes['is_a'] not in parent_edges:
                parent_edges.add(attributes['is_a'])
        return list(parent_edges)
    
    def get_nodes(self):
        nodes = {}
        for key, value in self.schema.items():
            if value['represented_as'] == 'node':
                if key in self.parent_nodes:
                    continue
                parent = value['is_a']
                currNode = {
                    'type': key,
                    'is_a': parent,
                    'label': value['input_label'],
                    'properties': value.get('properties', {})
                }
                if parent not in nodes:
                    nodes[parent] = []
                nodes[parent].append(currNode)

        return [{'child_nodes': nodes[key], 'parent_node': key} for key in nodes]

    def get_edges(self):
        edges = {}
        for key, value in self.schema.items():
            if value['represented_as'] == 'edge':
                if key in self.parent_edges:
                    continue
                label = value.get('output_lable', value['input_label'])
                edge = {
                    'type': key,
                    'label': label,
                    'is_a': value['is_a'],
                    'source': value.get('source', ''),
                    'target': value.get('target', ''),
                    'properties': value.get('properties', {})
                }
                parent = value['is_a']
                if parent not in edges:
                    edges[parent] = []
                edges[parent].append(edge)
        return [{'child_edges': edges[key], 'parent_edge': key} for key in edges]

    def get_relations_for_node(self, node):
        relations = []
        node_label = node.replace('_', ' ')
        for key, value in self.schema.items():
            if value['represented_as'] == 'edge':
                if 'source' in value and 'target' in value:
                    if value['source'] == node_label or value['target'] == node_label:
                        label = value.get('output_lable', value['input_label'])
                        relation = {
                            'type': key,
                            'label': label,
                            'source': value.get('source', ''),
                            'target': value.get('target', '')
                        }
                        relations.append(relation)
        return relations

    def get_schema():
        with open('schema_config.yaml', 'r') as file:
            prime_service = yaml.safe_load(file)

        schema = {}

        for key in prime_service.keys():
            if type(prime_service[key]) == str:
                continue
        
            if any(keys in prime_service[key].keys() for keys in ('source', 'target')):
                schema[key] = {
                    'source': prime_service[key]['source'],
                    'target': prime_service[key]['target']
                }

        return schema  
    
    def get_graph_info(self, file_path='./Data/graph_info.json'):
        try:
            with open(file_path, 'r') as file:
                graph_info = json.load(file)
                return graph_info
        except Exception as e:
            return {"error": str(e)} 

    def transform_to_graph_schema(self) -> Dict:
        """
        Transforms YAML schema into a format suitable for RequestIntermediator:
        {
            source_type: {
                target_type: [relationship_type]
            }
        }
        """
        graph_schema = {}
        
        # Process all edges from the schema
        for key, value in self.schema.items():
            if value.get('represented_as') == 'edge':
                source = value.get('source')
                target = value.get('target')
                
                # Handle both single targets and lists of targets
                if isinstance(target, list):
                    targets = target
                else:
                    targets = [target]
                
                # Initialize source in schema if not exists
                if source not in graph_schema:
                    graph_schema[source] = {}
                    
                # Add relationships for each target
                for t in targets:
                    if t not in graph_schema[source]:
                        graph_schema[source][t] = []
                    graph_schema[source][t].append(key)
                    
                    # Add inverse relationship if specified
                    if value.get('inverse'):
                        if t not in graph_schema:
                            graph_schema[t] = {}
                        if source not in graph_schema[t]:
                            graph_schema[t][source] = []
                        graph_schema[t][source].append(value['inverse'])
                        
        return graph_schema   
