import bpy
import math
from math import log10
from collections import defaultdict
import bmesh
import mathutils
import numpy as np
from bpy import context, data, ops
from mathutils.bvhtree import BVHTree
from mathutils import Vector
from mathutils.kdtree import KDTree
from mathutils import kdtree

def calculate_mesh_complexity(obj):
    # Complexity measured by the volume-to-surface area ratio
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    volume = bm.calc_volume()
    surface_area = sum(f.calc_area() for f in bm.faces)
    bm.free()

    return volume / surface_area if surface_area != 0 else 0

def calc_mesh_density(obj):
    total_area = sum(face.area for face in obj.data.polygons)
    n_vertices = len(obj.data.vertices)
    if total_area == 0: return n_vertices  # Avoid division by zero
    return n_vertices / total_area

def calculate_collapse_ratio(obj):
    
    # Calculate mesh density
    mesh_density = calc_mesh_density(obj)

    return max(min(300 / mesh_density, 0.5), 0.1)

def calculate_dissolve_angle_limit(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    angles = [face.normal.angle(other_face.normal) 
              for edge in bm.edges 
              if len(edge.link_faces) == 2
              for face, other_face in [(edge.link_faces[0], edge.link_faces[1])]
              if face.normal.length > 0 and other_face.normal.length > 0]

    bm.free()
    
    # Calculate average angle
    avg_angle = sum(angles) / len(angles) if angles else 0.0

    # Calculate standard deviation
    std_dev = np.std(angles) if angles else 0.0
    
    mesh_density = calc_mesh_density(obj)
    
    # Return the average angle multiplied by standard deviation
    return (avg_angle * (1 - std_dev))

def calculate_average_face_area_difference(obj):
    # Calculate face areas
    face_areas = np.array([face.area for face in obj.data.polygons])

    # Calculate differences between consecutive face areas
    differences = np.abs(np.diff(face_areas))

    # Calculate average difference
    average_difference = differences.mean() if differences.size else 0

    return average_difference

def decimate_objects(objects):
    for obj in objects:
        n_vertices = len(obj.data.vertices)
        
        # Create and configure decimation modifier
        decimate_modifier_name = "decimate_" + obj.name
        obj_mod = obj.modifiers.new(decimate_modifier_name, 'DECIMATE')

        # Choose decimation method based on the average different between face areas
        if calculate_average_face_area_difference(obj) < 5:
            obj_mod.decimate_type = 'COLLAPSE'
            obj_mod.ratio = calculate_collapse_ratio(obj)
        else:
            obj_mod.decimate_type = 'DISSOLVE'
            obj_mod.angle_limit = calculate_dissolve_angle_limit(obj)
            obj_mod.delimit = {'UV'}

def calculate_optimal_smoothing_angle(obj):

    # Get all the polygons of the mesh
    polygons = obj.data.polygons

    # Initialize angle list
    angles = []
    
    # Iterate over all the polygons in the mesh
    for poly in polygons:
        # Get all the vertices of the polygon
        vertices = poly.vertices

        # Only calculate angle if there are at least three vertices
        if len(vertices) >= 3:
            # Calculate and add up the angles between adjacent vertices
            for i in range(len(vertices)):
                # Get three consecutive vertices
                v1 = obj.data.vertices[vertices[i - 1]].co
                v2 = obj.data.vertices[vertices[i]].co
                v3 = obj.data.vertices[vertices[(i + 1) % len(vertices)]].co
                # Calculate the angle and add it to the total
                angle = (v2 - v1).angle(v2 - v3, 0)
                angles.append(angle)
    
    # Convert list of angles to numpy array for easier manipulation
    angles = np.array(angles)
    
    # Calculate the mean and standard deviation of the angles
    mean = np.mean(angles)
    std_dev = np.std(angles)
    
    # Use the mean and standard deviation to determine a suitable angle limit
    # Clip the angle limit between 0 and Pi/2
    angle_limit = np.clip(mean - std_dev * 0, 0, np.pi / 2)

    # Return the optimal angle limit
    return angle_limit

def calculate_surface_area(objects):
    areas = []
    for obj in objects:
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        area = sum(f.calc_area() for f in bm.faces)
        bm.free()
        areas.append(area)
    return areas


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def region_growing_clustering(objects, max_surface_area, d):
    areas = calculate_surface_area(objects)
    unvisited = set(range(len(objects)))
    clusters = []

    size = len(objects)
    kd = KDTree(size)
    
    for i, obj in enumerate(objects):
        kd.insert(obj.location, i)
    
    kd.balance()

    # Create BVH tree for each object
    bvhs = [BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get()) for obj in objects]

    while unvisited:
        seed = unvisited.pop()
        cluster = [seed]
        total_surface_area = areas[seed]

        frontier = [seed]
        while frontier:
            next_frontier = []

            for object_index in frontier:
                close_indices = sorted([(i, dist) for co, i, dist in kd.find_range(objects[object_index].location, d) if i in unvisited], key=lambda x: x[1])

                for i, dist in close_indices:
                    bvh = bvhs[i]

                    # Calculate the center of the bounding box of the other object
                    other_center = sum((Vector(b) for b in objects[i].bound_box), Vector()) / 8
                    
                    # Cast a ray from the current object's location to the other object's center
                    location, normal, index, distance = bvh.ray_cast(objects[object_index].location, other_center - objects[object_index].location)

                    # If the ray hit the other object (index is not None), the other object is inside the current one
                    # Check also if adding this object does not exceed the max_surface_area
                    if ((index is not None or dist <= d) and total_surface_area + areas[i] <= max_surface_area) and (i in unvisited):
                        cluster.append(i)
                        unvisited.remove(i)
                        next_frontier.append(i)
                        total_surface_area += areas[i]

            frontier = next_frontier

        clusters.append(cluster)

    return clusters


def clean_materials(obj):
    unique_materials = [slot.material for slot in obj.material_slots if slot.material not in unique_materials]
    for i, slot in enumerate(obj.material_slots):
        if slot.material in unique_materials[i+1:]:
            switch_material(obj, slot, unique_materials.index(slot.material))
            bpy.context.object.active_material_index = obj.material_slots.find(slot.name)
            bpy.ops.object.material_slot_remove()
    [remove_material_slot(obj, i) for i in reversed(range(len(obj.material_slots))) if not any(p.material_index == i for p in obj.data.polygons)]


def switch_material(obj, slot, other_slot_index):
    for poly in obj.data.polygons:
        if poly.material_index == slot.link:
            poly.material_index = other_slot_index


def remove_material_slot(obj, i):
    obj.active_material_index = i
    bpy.ops.object.material_slot_remove()


def merge_group(name, group, collection, max_surface_area, d):
    if len(group) == 0 or sum(area for area in calculate_surface_area(group)) == 0:
        return

    print(f"Calculating number of [{name}] clusters...")
    clusters = region_growing_clustering(group, max_surface_area, d)

    # Loop through each cluster
    for i, cluster_indices in enumerate(clusters):
        
        # Select objects that belong to the current cluster
        objs = [group[index] for index in cluster_indices]

        # Remove duplicate vertices from object
        remove_duplicate_vertices(objs)

        # Reduce number of faces
        apply_limited_dissolve(objs)

        if collection.name in collections_to_decimate:          
            # Deeply decimate objects
            decimate_objects(objs)

        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')

        # Apply all modifiers and select only the objects that are part of the current cluster
        for obj in objs:
            bpy.context.view_layer.objects.active = obj
            for modifier in obj.modifiers:
                bpy.ops.object.modifier_apply(modifier=modifier.name)
            obj.select_set(True)

        # Set the active object to the first object in the cluster
        bpy.context.view_layer.objects.active = objs[0]

        # Join all selected objects into one
        bpy.ops.object.join()

        # Remove unused material slots from the joined object
        if bpy.context.object.material_slots:
            # Create a bmesh object and fill it with the mesh data of the current object
            bm = bmesh.new()
            bm.from_mesh(bpy.context.object.data)

            # Get a set of all material indices used by faces in the mesh
            mats_used = {f.material_index for f in bm.faces}

            # Remove all unused material slots
            for slot_index in reversed(range(len(bpy.context.object.material_slots))):
                if slot_index not in mats_used:
                    bpy.context.object.active_material_index = slot_index
                    bpy.ops.object.material_slot_remove()

            # Free the bmesh object
            bm.free()
            
        # Optimize object's mesh
        remove_duplicate_vertices([objs[0]])

        # Switch back to object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Enable Auto Smooth and set the angle to an approximately optimal value
        bpy.ops.object.shade_smooth()
        bpy.context.object.data.use_auto_smooth = True
        bpy.context.object.data.auto_smooth_angle = calculate_optimal_smoothing_angle(bpy.context.object)
        
        # Attempt to fix inconsistent normals
        bpy.ops.object.modifier_add(type='WEIGHTED_NORMAL')
        bpy.context.object.modifiers["WeightedNormal"].keep_sharp = True

        # Rename the new joined object
        objs[0].name = f"{name}_cluster_{i}"

        # Print progress
        print(f'{name} [{i+1}/{len(clusters)}]: {round((i+1) / len(clusters) * 100)}%')

def remove_duplicate_vertices(objects):
    
    # Store the current mode
    original_mode = bpy.context.object.mode
    
    # Ensure we're in object mode
    if bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')
    
    # Set the object to be active and select it
    for obj in objects:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

    # Switch to edit mode
    if bpy.context.object.mode != "EDIT":
        bpy.ops.object.mode_set(mode='EDIT')

    # Remove duplicate vertices (merge vertices that occupy the same space)
    bpy.ops.mesh.remove_doubles(threshold=0.0002)

    # Delete loose geometry (isolated vertices, edges, and faces)
    bpy.ops.mesh.delete_loose()

    # Restore the original mode
    if bpy.context.object.mode != original_mode:
        bpy.ops.object.mode_set(mode=original_mode)

def calculate_average_face_area(obj):
    total_area = sum(face.area for face in obj.data.polygons)
    n_faces = len(obj.data.polygons)
    return total_area / n_faces if n_faces else 0

def apply_limited_dissolve(objects):
    
    # Store the current mode
    original_mode = bpy.context.object.mode
    
    # Ensure we're in object mode
    if bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')
    
    # Set the object to be active and select it
    entered_loop = False
    for obj in objects:
        if calculate_average_face_area(obj) > 500:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            entered_loop = True

    if entered_loop:
        # Switch to edit mode
        if bpy.context.object.mode != "EDIT":
            bpy.ops.object.mode_set(mode='EDIT')

        # Add Limited Dissolve operation
        bpy.ops.mesh.select_all(action='SELECT')  # Select all vertices
        bpy.ops.mesh.dissolve_limited(angle_limit=0.00087266, delimit={'UV'})  # Limit is in radians (approx. .05 degree here)

    # Restore the original mode
    if bpy.context.object.mode != original_mode:
        bpy.ops.object.mode_set(mode=original_mode)

def remove_faces_of_material(obj, material_name):
    
    # Gather indices of all material slots with the given material name
    material_indices = [i for i, slot in enumerate(obj.material_slots) if material_name in slot.material.name]

    # If there are no such indices, there's nothing to do
    if not material_indices:
        return

    # Create a new, empty bmesh
    bm = bmesh.new()

    # Fill the bmesh with the data from the object's mesh
    bm.from_mesh(obj.data)

    # Create a new integer layer for the faces of the bmesh
    material_layer = bm.faces.layers.int.new('temp_material')

    # Map each face to its material index
    for face in bm.faces:
        face[material_layer] = face.material_index

    # Generate a mapping from old to new indices, excluding those to be removed
    index_mapping = {i: j for j, i in enumerate(i for i in range(len(obj.material_slots)) if i not in material_indices)}

    # Iterate over a copy of the faces, so that removing faces doesn't interfere with the iteration
    for face in bm.faces[:]:
        # If this face has one of the unwanted materials, remove it
        if face[material_layer] in material_indices:
            bm.faces.remove(face)
        # Otherwise, remap its material index if needed
        elif face[material_layer] in index_mapping:
            face[material_layer] = index_mapping[face[material_layer]]

    # After all the removal, list all the face material indices
    face_material_indices = [face[material_layer] for face in bm.faces]

    # Create a new mesh data block and fill it with the data from the bmesh
    new_mesh_data = bpy.data.meshes.new('temp_data')
    bm.to_mesh(new_mesh_data)

    # Free the bmesh now that we're done with it
    bm.free()

    # Transfer material indices from the temporary layer to the actual material_index property
    for poly, material_index in zip(new_mesh_data.polygons, face_material_indices):
        poly.material_index = material_index

    # List all materials that are still used
    used_materials = [slot.material for i, slot in enumerate(obj.material_slots) if i not in material_indices]

    # Append all used materials to the new mesh
    for material in used_materials:
        new_mesh_data.materials.append(material)

    # Finally, replace the object's mesh with the new mesh
    obj.data = new_mesh_data


def get_collection_objects(collection):
    return [obj for obj in collection.objects if not (obj.animation_data or obj.find_armature() or obj.type != 'MESH')]

def main():
    for collection in bpy.data.collections:
        
        # Print the collection being processed
        print(f"Processing collection '{collection.name}'...")

        # Get all objects in the collection that are meshes and do not have any animation data or armature
        objects = get_collection_objects(collection)

        # Prepare to make objects single user
        print("Making objects and data single user...")
        bpy.ops.object.select_all(action='DESELECT')

        # Make each object a single user to avoid shared data
        for obj in objects:
            obj.select_set(True)
        bpy.ops.object.make_single_user(object=True, obdata=True)
        bpy.ops.object.select_all(action='DESELECT')
        
        # Initialize list to hold grouped objects
        grouped_objects = []

        # Remove invisible faces from each object
        for i, obj in enumerate(objects):
            remove_faces_of_material(obj, 'toolsnodraw')
            remove_faces_of_material(obj, 'toolsareaportal')
            print(f'Removing invisible faces [{i+1}/{len(objects)}]: {round((i+1) / len(objects) * 100)}%')

        print("Grouping objects based on material names...")
        groups = defaultdict(list)

        # Group objects based on their materials
        for name, properties in group_properties.items():
            for obj in objects:                    
                
                # Skip object if it's already grouped
                if obj in grouped_objects:
                    continue
                
                # Get materials of the object
                object_materials = {slot.material.name for slot in obj.material_slots if slot.material}

                 # Check if object's material matches group's materials and not in excluded materials
                matches_func = all if properties.get("match_all_materials", False) else any
                matches_materials = not "materials" in properties or matches_func(any(sub_string in mat for sub_string in properties["materials"]) for mat in object_materials) or not object_materials
                matches_excluded_materials = "excluded_materials" in properties and any(any(sub_string in mat for sub_string in properties["excluded_materials"]) for mat in object_materials) or not object_materials
                if matches_materials and not matches_excluded_materials:
                    
                    # Skip object if it doesn't contain all the substrings in object_names
                    if "object_names" in properties and not any(name_part in obj.name for name_part in properties["object_names"]):
                        continue
                    
                    # If required, remove the specified list of materials from the object
                    for material in (properties.get("materials_to_remove") or []):
                        remove_faces_of_material(obj, material)
                        print(f'Removing material "{material}" from object {obj.name}')
                    
                    # Mark object as processed
                    grouped_objects.append(obj)
                    
                    if "blacklisted" in properties and properties["blacklisted"]:
                        # Blacklist object
                        obj.name = f'{name}_{obj.name}'
                        
                    else:
                        # Add object to group
                        groups[name].append(obj)

        print("Merging and optimizing groups...")

        # Remove objects without any materials
        for obj in grouped_objects:
            if not obj.data.materials:
                print(f'Deleted object {obj.name} due to having no materials.')    
                bpy.data.objects.remove(obj, do_unlink=True)

        # Merge and optimize each group
        for name, group in groups.items():
            
            # Find merging properties of the group
            properties = group_properties[name]
            max_surface_area = properties.get("max_surface_area") or group_properties["Default"]["max_surface_area"]
            distance_between_objects = properties.get("distance_between_objects") or group_properties["Default"]["distance_between_objects"]
            
            # Merge all the objects in the group into a new cluster
            merge_group(name, group, collection, max_surface_area, distance_between_objects)
            print(f"Finished merging and optimizing group '{name}'")

        print("Deleting unused objects...")        

        # Print message when done processing the collection
        print(f"Finished processing collection '{collection.name}'")

    # Print message when done processing all collections
    print("Completed all tasks successfully.")


# Define group properties
group_properties = {

    "Cobweb": {"materials": ['cobweb'], 
              "match_all_materials": True, 
              "excluded_materials": [],
              "blacklisted": False,
              "distance_between_objects": 15},

    "Grass": {"materials": ['grass', 'shrub'], 
              "match_all_materials": False, 
              "excluded_materials": ['bark', 'trunk', 'log', 'blend'],
              "blacklisted": False,
              "distance_between_objects": 15},
    
    "Foliage": {"materials": ['materials/models/props_foliage', 'materials/models/props_forest/fir', 'materials/models/props_forest/pine'], 
                "match_all_materials": False, 
                "excluded_materials": [],
                "blacklisted": False,
                "distance_between_objects": 15},

    "Tracks": {"materials": ['materials/models/props_mining/track_mining'], 
               "match_all_materials": True, 
               "excluded_materials": [],
               "blacklisted": False,
               "max_surface_area": 512*512},

    "Terrain": {"materials": ['materials/nature'], 
                "match_all_materials": True, 
                "excluded_materials": [],
                "blacklisted": False,
                "distance_between_objects": 15,
                "max_surface_area": 2048*4096},

    "RemovedIllusionaries": {"materials": ['toolsblack'], 
                 "match_all_materials": False, 
                 "excluded_materials": [],
                 "blacklisted": True, 
                 "object_names": ['brush'],
                 "materials_to_remove": ['materials/tools/toolsblack']},
                 
    "RemovedDoorBarricades": {"materials": ['materials/metal/ibeam001b'], 
                 "object_names": ["func_door"],
                 "match_all_materials": True, 
                 "excluded_materials": [],
                 "blacklisted": True,
                 "materials_to_remove": ['materials/metal/ibeam001b']},
    
    "Gameplay": {"materials": ['materials/models/props_gameplay', 'door', 'billboard', 'cart_bomb'],
                 "match_all_materials": False, 
                 "excluded_materials": [],
                 "blacklisted": True},
                 
    "Dynamic": {"object_names": ['dynamic', 'physics'],
                "blacklisted": True},

    "Transparency": {"materials": ['chicken_wire', 'glass', 'metalgrate', 'cobweb'], 
                     "match_all_materials": False, 
                     "excluded_materials": [],
                     "blacklisted": False,
                     "distance_between_objects": 15},

    "Handrail": {"materials": ['handrail'], 
                 "match_all_materials": True, 
                 "excluded_materials": [],
                 "blacklisted": False},

    "Water": {"materials": ['materials/water'], 
              "match_all_materials": True, 
              "excluded_materials": [],
              "blacklisted": False,
              "distance_between_objects": 15},
              
    "Default": {"max_surface_area": 2048*1024,
                "distance_between_objects": 7}
}
collections_to_decimate = ["props"]

main()
