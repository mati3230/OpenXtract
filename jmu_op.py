import bpy
import sys
import numpy as np
from . import import_meshes
from . import mesh_operations

from bpy.types import Operator

# == GLOBAL VARIABLES
PLINKAGE_PROPS = [
    ('angle', bpy.props.IntProperty(name='Angle', default=90, min=0, max=180)),
    ('k_plinkage', bpy.props.IntProperty(name='k', default=100, min=5, max=150)),
    ('min_cluster_size', bpy.props.IntProperty(name='Min Cluster Size', default=10, min=2)),
    ('angle_dev', bpy.props.FloatProperty(name='Angle Deviation', default=10.0, min=0.0)),
    ('use_edges_plink', bpy.props.BoolProperty(name='Use Edges', default=False))
]

CP_PROPS = [
    ('k_cp', bpy.props.IntProperty(name='k', default=15, min=3, max=150)),
    ('reg_strength', bpy.props.FloatProperty(name='Reg Strength', default=0.3, min=0.001, max=10)),
    ('use_edges_cp', bpy.props.BoolProperty(name='Use Edges', default=False))
]

VCCS_PROPS = [
    ('voxel_resolution', bpy.props.FloatProperty(name='Voxel Resolution', default=0.5, min=0.0001, max=100)),
    ('seed_resolution', bpy.props.FloatProperty(name='Seed Resolution', default=0.5, min=0.0001, max=100)),
    ('color_importance', bpy.props.FloatProperty(name='Color Importance', default=0.3, min=0, max=1)),
    ('spatial_importance', bpy.props.FloatProperty(name='Spatial Importance', default=0.3, min=0, max=1)),
    ('normal_importance', bpy.props.FloatProperty(name='Normal Importance', default=0.3, min=0, max=1)),
    ('refinementIter', bpy.props.IntProperty(name='Refinement Iterations', default=0, min=0, max=5)),
    ('use_edges_vccs', bpy.props.BoolProperty(name='Use Edges', default=False)),
    ('r_search_gain_vccs', bpy.props.FloatProperty(name='Search Gain', default=0.5, min=0, max=1)),
]

VGS_PROPS = [
    ('voxel_size', bpy.props.FloatProperty(name='Voxel Size', default=0.5, min=0.0001, max=100)),
    ('graph_size', bpy.props.FloatProperty(name='Graph Size', default=1.5, min=0.0001, max=100)),
    ('sig_p', bpy.props.FloatProperty(name='Spatial Gain', default=0.2, min=0, max=1)),
    ('sig_n', bpy.props.FloatProperty(name='Angle Gain', default=0.2, min=0, max=1)),
    ('sig_o', bpy.props.FloatProperty(name='Stair Gain', default=0.2, min=0, max=1)),
    ('sig_e', bpy.props.FloatProperty(name='Eigen Gain', default=0.2, min=0, max=1)),
    ('sig_c', bpy.props.FloatProperty(name='Convex Gain', default=0.2, min=0, max=1)),
    ('sig_w', bpy.props.FloatProperty(name='Similarity Weight', default=2.0, min=0.001, max=100)),
    ('cut_thred', bpy.props.FloatProperty(name='Cut Threshold', default=0.3, min=0.001, max=1)),
    ('points_min', bpy.props.IntProperty(name='Min Points', default=0, min=0, max=1000000)),
    ('adjacency_min', bpy.props.IntProperty(name='Min Adjacency', default=0, min=0, max=1000)),
    ('voxels_min', bpy.props.IntProperty(name='Min Voxels', default=0, min=0, max=1000)),
    #
    ('seed_size', bpy.props.FloatProperty(name='Seed Size', default=0.3, min=0.0001, max=100)),
    ('sig_f', bpy.props.FloatProperty(name='f Gain', default=0.2, min=0, max=1)),
    ('sig_a', bpy.props.FloatProperty(name='a Gain', default=0.0, min=0, max=1)),
    ('sig_b', bpy.props.FloatProperty(name='b Gain', default=0.25, min=0, max=1)),
    ('use_edges_vgs', bpy.props.BoolProperty(name='Use Edges', default=False)),
    ('r_search_gain_svgs', bpy.props.FloatProperty(name='Search Gain', default=0.5, min=0, max=1)),
]

RANSAC_PROPS = [
    ('distance_threshold', bpy.props.FloatProperty(name='Distance Threshold', default=0.1, min=0.0001, max=100)),
    ('num_iterations', bpy.props.IntProperty(name='N Iterations', default=1000, min=10, max=10000)),
]

class JMU_OT_Simplify(Operator):
    bl_idname = "object.simplyfy"
    bl_label = "Adds a Decimate Modifier"
    bl_description = "Simplify" 
    def invoke(self, context, event):
        sel = bpy.context.selected_objects
        act = bpy.context.active_object

        for obj in sel:
            bpy.context.view_layer.objects.active = obj #sets the obj accessible to bpy.ops
            bpy.ops.object.modifier_add(type='DECIMATE')
            bpy.context.object.modifiers["Decimate"].ratio = 0.9

        bpy.context.view_layer.objects.active = act

        return {'FINISHED'}
    def execute(self, context):
        return {'FINISHED'} 


class JMU_OT_Combine_Superpoints(Operator):
    bl_idname = "object.combine_superpoints"
    bl_label = "Combine Superpoints"
    bl_description = "Combines superpoints into one cluster" 
    def invoke(self, context, event):
        #bpy.ops.object.select_all(action='DESELECT')

        #mesh = [m for m in bpy.context.scene.objects if m.type == 'MESH']

        # for obj in mesh:
        #     obj.select_set(state=True)

        #    bpy.context.view_layer.objects.active = obj
        # bpy.context.view_layer.objects.active = obj #sets the obj accessible to bpy.ops
        bpy.ops.object.join()
        return {'FINISHED'}
    def execute(self, context):
        return {'FINISHED'}  


class JMU_OT_Reduce_Superpoint(Operator):
    bl_idname = "object.reduce_superpoint"
    bl_label = "Reduce Superpoint"
    bl_description = "Reduces Points from a Superpoint" 
    def invoke(self, context, event):
        # bpy.ops.object.mode_set(mode='EDIT')        
        bpy.ops.mesh.separate(type='SELECTED')
        bpy.ops.object.mode_set(mode='OBJECT')
        return {'FINISHED'}
    def execute(self, context):
        return {'FINISHED'}  


class JMU_OT_Run_Sem_Seg(Operator):
    bl_idname = "object.run_sem_seg"
    bl_label = "Run"
    bl_description = "Semantic Segmentation of the Active Object" 
    def invoke(self, context, event):
        mesh_operations.semantic_segmentation(context)   
        return {'FINISHED'}
    def execute(self, context):
        return {'FINISHED'}  


class JMU_OT_Triangulate(Operator):
    bl_idname = "object.triangulate"
    bl_label = "Triangulate"
    bl_description = "Adds a Triangulate Modifier" 
    def invoke(self, context, event):
        sel = bpy.context.selected_objects
        act = bpy.context.active_object

        for obj in sel:
            bpy.context.view_layer.objects.active = obj #sets the obj accessible to bpy.ops
            bpy.ops.object.modifier_add(type='TRIANGULATE')


        bpy.context.view_layer.objects.active = act

        return {'FINISHED'}
        return {'FINISHED'}
    def execute(self, context):
        return {'FINISHED'}  


class JMU_OT_Run_PLinkage_Op(Operator):
    bl_idname = "object.run_plinkage"
    bl_label = "Run"
    bl_description = "Partition of the active Object"

    def invoke(self, context, event):
        params = (
                context.scene.angle,
                context.scene.k_plinkage,
                context.scene.min_cluster_size,
                context.scene.angle_dev,
                context.scene.use_edges_plink
            )

        mesh_operations.apply_plinkage(context, params)   
            
        print("Finished")
        return {'FINISHED'}

    def execute(self, context):
         return {'FINISHED'}   


class JMU_OT_Run_CP_Op(Operator):
    bl_idname = "object.run_cp"
    bl_label = "Run"
    bl_description = "Partition of the active Object"

    def invoke(self, context, event):
        params = (
                context.scene.k_cp,
                context.scene.reg_strength,
                context.scene.use_edges_cp
            )

        mesh_operations.apply_cp(context, params)   
            
        print("Finished")
        return {'FINISHED'}

    def execute(self, context):
         return {'FINISHED'}


class JMU_OT_Run_VCCS_Op(Operator):
    bl_idname = "object.run_vccs"
    bl_label = "Run"
    bl_description = "Partition of the active Object"

    def invoke(self, context, event):
        params = (
            context.scene.voxel_resolution,
            context.scene.seed_resolution,
            context.scene.color_importance,
            context.scene.spatial_importance,
            context.scene.normal_importance,
            context.scene.refinementIter,
            context.scene.use_edges_vccs,
            context.scene.r_search_gain_vccs
        )

        mesh_operations.apply_vccs(context, params)   
            
        print("Finished")
        return {'FINISHED'}

    def execute(self, context):
         return {'FINISHED'}


class JMU_OT_Run_VGS_Op(Operator):
    bl_idname = "object.run_vgs"
    bl_label = "Run"
    bl_description = "Partition of the active Object"

    def invoke(self, context, event):
        params = (
            context.scene.voxel_size,
            context.scene.graph_size,
            context.scene.sig_p,
            context.scene.sig_n,
            context.scene.sig_o,
            context.scene.sig_e,
            context.scene.sig_c,
            context.scene.sig_w,
            context.scene.cut_thred,
            context.scene.points_min,
            context.scene.adjacency_min,
            context.scene.voxels_min,
            #
            context.scene.seed_size,
            context.scene.sig_f,
            context.scene.sig_a,
            context.scene.sig_b,
            context.scene.use_edges_vgs,
            context.scene.r_search_gain_sgvs
        )

        mesh_operations.apply_vgs(context, params)   
            
        print("Finished")
        return {'FINISHED'}

    def execute(self, context):
         return {'FINISHED'}


class JMU_OT_Run_SVGS_Op(Operator):
    bl_idname = "object.run_svgs"
    bl_label = "Run"
    bl_description = "Partition of the active Object"
    
    def invoke(self, context, event):
        params = (
            context.scene.voxel_size,
            context.scene.graph_size,
            context.scene.sig_p,
            context.scene.sig_n,
            context.scene.sig_o,
            context.scene.sig_e,
            context.scene.sig_c,
            context.scene.sig_w,
            context.scene.cut_thred,
            context.scene.points_min,
            context.scene.adjacency_min,
            context.scene.voxels_min,
            #
            context.scene.seed_size,
            context.scene.sig_f,
            context.scene.sig_a,
            context.scene.sig_b,
            context.scene.use_edges_vgs,
            context.scene.r_search_gain_svgs
        )

        mesh_operations.apply_svgs(context, params)   
            
        print("Finished")
        return {'FINISHED'}

    def execute(self, context):
         return {'FINISHED'}


class JMU_OT_Run_RANSAC_Op(Operator):
    bl_idname = "object.run_ransac"
    bl_label = "Run"
    bl_description = "Partition of the active Object"
    
    def invoke(self, context, event):
        params = (
            context.scene.distance_threshold,
            context.scene.num_iterations
        )

        mesh_operations.apply_ransac(context, params)   
            
        print("Finished")
        return {'FINISHED'}

    def execute(self, context):
         return {'FINISHED'}


@classmethod
def poll(cls, context):
    obj = context.object
    if obj is not None:
            if obj.mode == "OBJECT":
                return True
    return False
