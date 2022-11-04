import bpy

def import_custom_mesh(file_loc):
    ## todo: not working
    file_loc = './ExampleMeshes/boulder.glb'
    imported_object = bpy.ops.import_mesh.ply(filepath=file_loc)
    return

def import_suzanne():
    bpy.ops.mesh.primitive_monkey_add()
    return