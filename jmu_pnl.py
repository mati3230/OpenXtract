import bpy

from bpy.types import Panel
from .jmu_op import PLINKAGE_PROPS, CP_PROPS, VCCS_PROPS, VGS_PROPS, RANSAC_PROPS
SEMSEG=True
try:
    import torch
except:
    SEMSEG=False

def add_algorithm(context, row, col, text, buttons_dict, props):
    row = col.row()
    col.label(text=text)

    for k, v in buttons_dict.items():
        col.operator(k, text=v)
    
    row = col.row()
    i = 0
    for (prop_name, _) in props:
        row.prop(context.scene, prop_name)
        i += 1
        if (i % 3 == 0) and (i < len(props)):
            row = col.row()
    return row, col

class JMU_PT_Panel(Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_label = "OpenXtract"
    bl_category = "OpenXtract"

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        col = row.column()       
        
        row, col = add_algorithm(context=context, row=row, col=col,
            text='Plane Partition',
            buttons_dict={
                "object.run_ransac": "Run"
            },
            props=RANSAC_PROPS)

        row, col = add_algorithm(context=context, row=row, col=col,
            text='P Linkage (Partition)',
            buttons_dict={
                "object.run_plinkage": "Run"
            },
            props=PLINKAGE_PROPS)

        row, col = add_algorithm(context=context, row=row, col=col,
            text='Cut Pursuit (Partition)',
            buttons_dict={
                "object.run_cp": "Run"
            },
            props=CP_PROPS)

        row, col = add_algorithm(context=context, row=row, col=col,
            text='VCCS (Partition)',
            buttons_dict={
                "object.run_vccs": "Run"
            },
            props=VCCS_PROPS)

        row, col = add_algorithm(context=context, row=row, col=col,
            text='VGS & SVGS (Partition)',
            buttons_dict={
                "object.run_vgs": "Run VGS",
                "object.run_svgs": "Run SVGS"
            },
            props=VGS_PROPS)

        if SEMSEG:
            row = col.row()
            col.label(text='PointNet++ (Semantic Segmentation)')
            col.operator("object.run_sem_seg", text="Run")

        row = col.row()
        col.label(text='Superpoint Operations')
        col.operator("object.combine_superpoints", text="Combine")
        col.operator("object.reduce_superpoint", text="Reduce")

        row = col.row()
        col.label(text='Modifiers')
        col.operator("object.triangulate", text="Triangulate")
        col.operator("object.simplyfy", text="Decimate")