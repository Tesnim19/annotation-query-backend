from typing import Dict, List, Set
from app import schema_manager

class RequestIntermediator:
    def __init__(self):
        self.schema = schema_manager.transform_to_graph_schema()

    def find_all_paths(self, start_type: str, end_type: str, max_depth: int = 5) -> List[List[str]]:
        """
        Find all possible paths between two node types using the schema.
        First checks for direct relationship, then looks for paths with intermediaries.
        Returns tuple: (has_direct_path, all_paths)
        """
        paths = []
        
        # Check for direct relationship first
        direct_type = self.get_relationship_type(start_type, end_type)
        has_direct_path = direct_type is not None
        if has_direct_path:
            paths.append([start_type, end_type])  # Direct path
        
        # Then look for paths with intermediaries
        def dfs(current_type: str, target_type: str, visited: Set[str], path: List[str]) -> List[List[str]]:
            if len(path) > max_depth:
                return []
                
            if current_type == target_type and len(path) > 2:  # Only paths with intermediaries
                return [path[:]]
                
            intermediate_paths = []
            if current_type in self.schema:
                for next_type in self.schema[current_type].keys():
                    if next_type not in visited:
                        visited.add(next_type)
                        new_paths = dfs(next_type, target_type, visited, path + [next_type])
                        intermediate_paths.extend(new_paths)
                        visited.remove(next_type)
            return intermediate_paths

        # Add paths with intermediaries
        intermediate_paths = dfs(start_type, end_type, {start_type}, [start_type])
        paths.extend(intermediate_paths)
        
        return has_direct_path, paths

    def get_relationship_type(self, source_type: str, target_type: str) -> str:
        """Get the relationship type between two node types from schema"""
        if source_type in self.schema and target_type in self.schema[source_type]:
            # Get the full key (e.g., 'gene-expressed_in-clo')
            full_key = self.schema[source_type][target_type][0]
            # Extract just the relationship type (e.g., 'expressed_in')
            # by splitting on '-' and taking the middle part
            parts = full_key.split('-')
            if len(parts) == 3:
                return parts[1]
        return None

    def enhance_request(self, request: Dict) -> List[Dict]:
        """
        Enhance the original request by handling all unspecified predicates,
        generating all possible combinations of paths.
        """
        base_request = {
            'nodes': request.get('nodes', []).copy(),
            'predicates': request.get('predicates', []).copy()
        }
        
        # Find all unspecified predicates
        unspecified_predicates = [
            (i, p) for i, p in enumerate(base_request['predicates']) 
            if not p.get('type')
        ]
        
        if not unspecified_predicates:
            return [request]

        # Check if we need to suggest paths
        needs_suggestions = False
        for pred_index, predicate in unspecified_predicates:
            source_node = next(n for n in base_request['nodes'] if n['node_id'] == predicate['source'])
            target_node = next(n for n in base_request['nodes'] if n['node_id'] == predicate['target'])
            
            has_direct, paths = self.find_all_paths(source_node['type'], target_node['type'])
            
            if not has_direct and not paths:
                # No valid paths at all
                raise ValueError(f"No valid paths found between {source_node['type']} and {target_node['type']}")
                
            if len(paths) > 1 or (not has_direct and paths):
                # We have multiple paths or only indirect paths
                needs_suggestions = True
                break

        if not needs_suggestions:
            # Just fill in direct relationships and return
            enhanced = base_request.copy()
            for pred_index, predicate in unspecified_predicates:
                source_node = next(n for n in enhanced['nodes'] if n['node_id'] == predicate['source'])
                target_node = next(n for n in enhanced['nodes'] if n['node_id'] == predicate['target'])
                rel_type = self.get_relationship_type(source_node['type'], target_node['type'])
                enhanced['predicates'][pred_index]['type'] = rel_type
            return [enhanced]

        # If we reach here, we need to generate path suggestions
        all_path_options = []
        for pred_index, predicate in unspecified_predicates:
            source_node = next(n for n in base_request['nodes'] if n['node_id'] == predicate['source'])
            target_node = next(n for n in base_request['nodes'] if n['node_id'] == predicate['target'])
            
            _, paths = self.find_all_paths(source_node['type'], target_node['type'])
            all_path_options.append({
                'pred_index': pred_index,
                'predicate': predicate,
                'paths': paths
            })

        # Generate all combinations of paths
        enhanced_requests = [base_request]
        for path_info in all_path_options:
            new_requests = []
            for existing_request in enhanced_requests:
                for path in path_info['paths']:
                    enhanced = self._create_enhanced_request(
                        existing_request.copy(),
                        path_info['pred_index'],
                        path_info['predicate'],
                        path
                    )
                    if enhanced:
                        new_requests.append(enhanced)
            
            if new_requests:
                enhanced_requests = new_requests

        return enhanced_requests

    def _create_enhanced_request(self, base_request: Dict, pred_index: int, original_predicate: Dict, path: List[str]) -> Dict:
        """
        Creates an enhanced request by adding intermediate nodes and relationships for a given path
        """
        enhanced_request = {
            'nodes': base_request['nodes'].copy(),
            'predicates': base_request['predicates'].copy()
        }
        
        # Get the next available node ID
        next_node_id = max([int(node['node_id'][1:]) for node in enhanced_request['nodes']]) + 1
        
        # Track the current source node ID
        current_node_id = original_predicate['source']
        new_predicates = []
        
        # Add intermediate nodes and relationships
        for i in range(1, len(path) - 1):
            new_node_id = f"n{next_node_id}"
            next_node_id += 1
            
            # Add new intermediate node
            new_node = {
                "node_id": new_node_id,
                "id": "",
                "type": path[i],
                "properties": {}
            }
            enhanced_request['nodes'].append(new_node)
            
            # Add relationship from previous node to new node
            rel_type = self.get_relationship_type(
                next(n['type'] for n in enhanced_request['nodes'] if n['node_id'] == current_node_id),
                path[i]
            )
            if not rel_type:
                return None
                
            new_predicates.append({
                "type": rel_type,
                "source": current_node_id,
                "target": new_node_id
            })
            current_node_id = new_node_id
        
        # Add final relationship to target
        final_rel_type = self.get_relationship_type(
            next(n['type'] for n in enhanced_request['nodes'] if n['node_id'] == current_node_id),
            next(n['type'] for n in enhanced_request['nodes'] if n['node_id'] == original_predicate['target'])
        )
        if not final_rel_type:
            return None
            
        new_predicates.append({
            "type": final_rel_type,
            "source": current_node_id,
            "target": original_predicate['target']
        })
        
        # Replace the original predicate with the new chain of predicates
        enhanced_request['predicates'] = [
            pred if i != pred_index else new_predicates[0]
            for i, pred in enumerate(enhanced_request['predicates'])
        ]
        enhanced_request['predicates'].extend(new_predicates[1:])
        
        return enhanced_request

    def process_request(self, request: Dict) -> Dict:
        """Main entry point to process and enhance a request"""
        enhanced_requests = self.enhance_request(request)
        
        # If we have multiple enhanced requests, format them for selection
        if len(enhanced_requests) > 1:
            return {
                "status": "needs_enhancement",
                "possible_paths": {
                    f"request_{i+1}": enhanced_req
                    for i, enhanced_req in enumerate(enhanced_requests)
                }
            }
        # If we only have one request (either original or single enhancement), return it
        return enhanced_requests[0]