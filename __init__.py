# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name" : "OpenXtract",
    "author" : "Marcel Tiator, Calvin Huhn, Christian Geiger, Paul Grimm",
    "description" : "",
    "blender" : (2, 93, 6),
    "version" : (0, 0, 1),
    "location" : "View3D",
    "warning" : "",
    "category" : "Object"
}
import bpy
from .jmu_op import PLINKAGE_PROPS
from .jmu_op import CP_PROPS
from .jmu_op import VCCS_PROPS
from .jmu_op import VGS_PROPS
from .jmu_op import RANSAC_PROPS

from .jmu_op import JMU_OT_Run_RANSAC_Op
from .jmu_op import JMU_OT_Run_PLinkage_Op
from .jmu_op import JMU_OT_Run_CP_Op
from .jmu_op import JMU_OT_Run_VCCS_Op
from .jmu_op import JMU_OT_Run_VGS_Op
from .jmu_op import JMU_OT_Run_SVGS_Op
from .jmu_op import JMU_OT_Combine_Superpoints
from .jmu_op import JMU_OT_Reduce_Superpoint
from .jmu_op import JMU_OT_Run_Sem_Seg
from .jmu_op import JMU_OT_Triangulate
from .jmu_op import JMU_OT_Simplify

from .jmu_pnl import JMU_PT_Panel
classes = (JMU_OT_Run_RANSAC_Op, JMU_OT_Run_PLinkage_Op, JMU_OT_Simplify, JMU_OT_Run_CP_Op, JMU_OT_Run_VCCS_Op, JMU_OT_Run_VGS_Op, JMU_OT_Run_SVGS_Op, JMU_PT_Panel, JMU_OT_Combine_Superpoints, JMU_OT_Reduce_Superpoint, JMU_OT_Run_Sem_Seg, JMU_OT_Triangulate)

def register_props(props):
    for (prop_name, prop_value) in props:
        setattr(bpy.types.Scene, prop_name, prop_value)

def unregister_props(props):
    for (prop_name, _) in props:
        delattr(bpy.types.Scene, prop_name)

# loaded
def register():
    register_props(props=RANSAC_PROPS)
    register_props(props=PLINKAGE_PROPS)
    register_props(props=CP_PROPS)
    register_props(props=VCCS_PROPS)
    register_props(props=VGS_PROPS)
    for c in classes:
        bpy.utils.register_class(c)

#unloaded
def unregister():
    unregister_props(props=RANSAC_PROPS)
    unregister_props(props=PLINKAGE_PROPS)
    unregister_props(props=CP_PROPS)
    unregister_props(props=VCCS_PROPS)
    unregister_props(props=VGS_PROPS)
    for c in classes:
        bpy.utils.unregister_class(c)
