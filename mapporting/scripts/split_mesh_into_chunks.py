import bpy
import bmesh
from mathutils import Vector

# Get the active object
obj = bpy.context.active_object

# Check if the object is a mesh
if obj and obj.type == 'MESH':
    # Get the bounding box of the object
    min_bound = Vector(obj.bound_box[0])
    max_bound = Vector(obj.bound_box[6])
    size = max_bound - min_bound
    
    # Set the size of each cube chunk (from the provided file, using 15 units)
    chunk_size = 15.0  # Adjust this based on your specific needs
    
    # Calculate the number of chunks along each axis
    num_chunks = Vector((
        int(size.x / chunk_size) + 1,
        int(size.y / chunk_size) + 1,
        int(size.z / chunk_size) + 1
    ))
    
    print(f"Object: {obj.name}")
    print(f"Chunk Size: {chunk_size}")
    print(f"Number of Chunks (approx): {num_chunks}")
    
    # Enter Edit Mode and use BMesh
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    
    # Deselect all faces
    bpy.ops.mesh.select_all(action='DESELECT')
    
    # Create a dictionary to hold the chunks
    chunks = {}
    
    for face in bm.faces:
        # Get the center point of the face
        face_center = face.calc_center_median()
        
        # Determine the chunk index for the face based on its position
        chunk_index = (
            int((face_center.x - min_bound.x) / chunk_size),
            int((face_center.y - min_bound.y) / chunk_size),
            int((face_center.z - min_bound.z) / chunk_size)
        )
        
        # Add the face to the corresponding chunk
        if chunk_index not in chunks:
            chunks[chunk_index] = []
        chunks[chunk_index].append(face)
    
    chunk_count = 0
    for chunk_index, chunk_faces in chunks.items():
        chunk_count += 1
        chunk_name = f"Chunk_{chunk_count:03d}"
        
        print(f"Processing Chunk {chunk_count}: Index {chunk_index}")
        print(f" - Number of Faces: {len(chunk_faces)}")
        
        # Deselect all faces
        bpy.ops.mesh.select_all(action='DESELECT')
        
        # Select faces in the current chunk
        for face in chunk_faces:
            face.select = True
        
        # Separate the selected faces into a new object
        bpy.ops.mesh.separate(type='SELECTED')
        
        # Rename the new chunk object for clarity
        new_chunk = bpy.context.selected_objects[-1]
        new_chunk.name = chunk_name
        print(f" - New Object: {new_chunk.name}")
    
    # Update the mesh and switch back to Object Mode
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    print("Chunking completed.")
    
else:
    print("Please select a mesh object.")