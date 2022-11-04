from logging import logProcesses
import bpy
import bmesh
import importlib
SEMSEG=True
try:
    import torch
except:
    SEMSEG=False
import os
import sys
import numpy as np
import pyransac3d as pyrsc

from . import libplink
from . import libgeo
from . import libply_c
from . import libcp
from . import libvccs
from . import libvgs

from . import graph_utils
from . import utils
import mathutils
import random
if SEMSEG:
    from . import pointnet2_sem_seg
    from .data_utils.S3DISDataLoader import CustomScene


def get_verts(obj):
    if obj.mode == 'EDIT':
        # this works only in edit mode,
        bm = bmesh.from_edit_mesh(obj.data)
        verts = [vert.co for vert in bm.verts]
    else:
        # this works only in object mode,
        verts = [vert.co for vert in obj.data.vertices]
        #print(len(obj.data.edges))
    return [vert.to_tuple() for vert in verts]


def get_normals(obj):
    if obj.mode == 'EDIT':
        # this works only in edit mode,
        bm = bmesh.from_edit_mesh(obj.data)
        normals = [vert.normal for vert in bm.verts]
    else:
        # this works only in object mode,
        normals = [vert.normal for vert in obj.data.vertices]
        #print(len(obj.data.edges))
    return np.array(normals, dtype=np.float32)


def get_graph(context):
    xyz, rgb, P, n_points, obj, colors_avaible, normals = get_cloud(context=context, return_normals=True)
    adjacency_list = get_adjacency_list(obj=obj, P=P)
    source, target, distances, uni_verts, direct_neigh_idxs, n_edges = graph_utils.get_edges(
        mesh_vertices=xyz, adj_list=adjacency_list)

    source = source.astype(np.uint32)
    target = target.astype(np.uint32)
    uni_verts = uni_verts.astype(np.uint32)
    direct_neigh_idxs = direct_neigh_idxs.astype(np.uint32)
    n_edges = n_edges.astype(np.uint32)
    distances = distances.astype(np.float32)
    return xyz, rgb, P, n_points, obj, colors_avaible, normals,\
        adjacency_list,\
        source, target, uni_verts, direct_neigh_idxs, n_edges, distances


def get_vertex_colors(obj, n_points):
    # lookup in indices list makes this method really slow!!!
    me = obj.data
    indices = []
    rgb = np.zeros((n_points, 3), dtype=np.float32)
    for i in range(len(me.polygons)):
        for li in me.polygons[i].loop_indices:
            index = me.loops[li].vertex_index
            if index in indices:
                continue
            indices.append(index)
            try: # me.vertex_colors.active is NoneType if mesh has no colors
                color = me.vertex_colors.active.data[li].color    
            except:
                return rgb
            rgb[index] = np.array([color[0], color[1], color[2]])
    return rgb


def get_vertex_colors2(obj, n_points):
    me = obj.data
    rgb = np.zeros((n_points, 3), dtype=np.float32)
    colors_avaible = True
    for i in range(len(me.polygons)):
        for li in me.polygons[i].loop_indices:
            index = me.loops[li].vertex_index
            try: # me.vertex_colors.active is NoneType if mesh has no colors
                color = me.vertex_colors.active.data[li].color    
            except:
                colors_avaible = False
                return rgb, colors_avaible
            rgb[index] = np.array([color[0], color[1], color[2]])
    return rgb, colors_avaible


def set_vertex_colors(mesh, vidx_2_col):
    mesh.vertex_colors.new()
    color_layer = mesh.vertex_colors["Col"]
    for i in range(len(mesh.polygons)):
        for li in mesh.polygons[i].loop_indices:
            vertex_index = mesh.loops[li].vertex_index
            color_layer.data[li].color = vidx_2_col[vertex_index]


def get_bmesh(verts, faces):
    bm = bmesh.new()

    for v_co in verts:
        bm.verts.new(v_co)

    bm.verts.ensure_lookup_table()
    for f_idx in faces:
        try:
            bm.faces.new([bm.verts[i] for i in f_idx])
        except:
            continue
    return bm


def get_face_from_polygon(mesh, polygon):
    face = []
    for loop_index in range(polygon.loop_start, polygon.loop_start + polygon.loop_total):
        face.append(mesh.loops[loop_index].vertex_index)
    return face


def create_meshes(p_vec, xyz, rgb, obj, classes=None, colors_avaible=True, algorithm="collection"):
    me = obj.data
    sp_2_verts, sp_2_tris, sp_2_cols = meshes_from_superpoints(mesh=me, p_vec=p_vec, xyz=xyz, rgb=rgb)
    
    collection = bpy.data.collections.new(algorithm)
    bpy.context.scene.collection.children.link(collection)
    
    for key, value in sp_2_verts.items():
        verts = sp_2_verts[key]
        faces = sp_2_tris[key]

        name = str(key)
        if classes is not None:
            name = classes[key]
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)  # add a new object using the mesh
        collection.objects.link(obj)

        bm = get_bmesh(verts=verts, faces=faces)
        bm.to_mesh(mesh)
        mesh.update()
        bm.free()
        if colors_avaible:
            set_vertex_colors(mesh=mesh, vidx_2_col=sp_2_cols[key])


def meshes_from_superpoints(mesh, p_vec, xyz, rgb):
    sp_2_tri = {}
    sp_2_verts = {}
    sp_2_v_index = {}
    sp_2_vert_2_index = {}
    sp_2_cols = {}
    uni_p_vec, counts_p_vec = np.unique(p_vec, return_counts=True)
    for sp in uni_p_vec:
        sp_2_tri[sp] = []
        sp_2_verts[sp] = []
        sp_2_v_index[sp] = 0
        sp_2_vert_2_index[sp] = {}
        sp_2_cols[sp] = {}
    for i in range(len(mesh.polygons)):
        polygon = mesh.polygons[i]
        face = get_face_from_polygon(mesh=mesh, polygon=polygon)
        np_face = np.array(face, dtype=np.uint32)
        sp_labels = p_vec[np_face] # superpoint labels of that face
        
        sp_labels = np.unique(sp_labels)
        sp_sizes = []
        for j in range(sp_labels.shape[0]):
            sp_label = sp_labels[j]
            idx = np.where(uni_p_vec == sp_label)[0]
            sp_size = counts_p_vec[idx]
            sp_sizes.append(sp_size)

        max_idx = np.argmax(sp_sizes)
        tri_label = sp_labels[max_idx]
        
        face_points = xyz[np_face]
        face_colors = rgb[np_face]
        new_face = []
        for j in range(face_points.shape[0]):
            v = tuple(face_points[j].tolist())
            #new_face.append(sp_2_v_index[tri_label])
            if v in sp_2_vert_2_index[tri_label]:
                v_index = sp_2_vert_2_index[tri_label][v]
            else:
                sp_2_verts[tri_label].append(v)
                v_index = sp_2_v_index[tri_label]
                sp_2_vert_2_index[tri_label][v] = v_index
                sp_2_v_index[tri_label] += 1
            new_face.append(v_index)
            sp_2_cols[tri_label][v_index] = tuple(face_colors[j].tolist()) + (1.0, )

        face = tuple(new_face)
        
        sp_2_tri[tri_label].append(face)
    return sp_2_verts, sp_2_tri, sp_2_cols


def get_cloud(context, return_normals=False):
    active_obj = context.view_layer.objects.active
    obj = bpy.context.active_object
    
    # coordinates as tuples
    plain_verts = get_verts(obj)

    xyz = np.array(plain_verts, copy=True)
    rgb, colors_avaible = get_vertex_colors2(obj=obj, n_points=xyz.shape[0])
    P = np.hstack((xyz, rgb))
    P = P.astype(np.float32)
    n_points = P.shape[0]
    if return_normals:
        normals = get_normals(obj)
        return xyz, rgb, P, n_points, obj, colors_avaible, normals
    else:
        return xyz, rgb, P, n_points, obj, colors_avaible


def get_adjacency_list(obj, P):
    edges = obj.data.edges
    edgesList = []
    for i in range(len(edges)):
        edgesList.append([edges[i].vertices[0], edges[i].vertices[1]])  
    adjacency_list = utils.compute_adjacency(P, edgesList)
    return adjacency_list


def apply_plinkage(context, params):
    (angle, k, min_cluster_size, angle_dev, use_edges) = params
    
    if use_edges:
        xyz, rgb, P, n_points, obj, colors_avaible, normals,\
        adjacency_list,\
        source, target, uni_verts, direct_neigh_idxs, n_edges, distances = get_graph(context=context)

        exclude_closest = False
        source_, target_, distances_, ok = libgeo.geodesic_knn(uni_verts, direct_neigh_idxs, n_edges, target, distances, k, exclude_closest)

        if not ok[0]:
            print("Error while searching geodesic neighbourhood - apply point cloud p-linkage")
            point_idxs, p_vec, duration = libplink.plinkage(P, k=k, angle=angle,
                min_cluster_size=min_cluster_size, angle_dev=angle_dev)
        else:
            use_normals = False
            point_idxs, p_vec, duration = libplink.plinkage_geo(P, target_, normals, k=k, angle=angle, min_cluster_size=min_cluster_size, angle_dev=angle_dev, use_normals=use_normals)
    else:
        xyz, rgb, P, n_points, obj, colors_avaible = get_cloud(context=context)
        point_idxs, p_vec, duration = libplink.plinkage(P, k=k, angle=angle, min_cluster_size=min_cluster_size,
                                                     angle_dev=angle_dev)
        adjacency_list = get_adjacency_list(obj=obj, P=P)
    _, p_vec = graph_utils.refine(point_idxs=point_idxs, p_vec=p_vec, adjacency_list=adjacency_list)
    if sys.platform == "darwin": # apple
         p_vec = np.array(p_vec, dtype=np.int32)    
    create_meshes(p_vec=p_vec, xyz=xyz, rgb=rgb, obj=obj, colors_avaible=colors_avaible, algorithm="p-linkage")


def unidirectional(graph_nn=None):
    source = graph_nn["source"]
    target = graph_nn["target"]
    distances = graph_nn["distances"]
    uni_verts, direct_neigh_idxs, n_edges = np.unique(source, return_index=True, return_counts=True)
    uni_verts = uni_verts.astype(np.uint32)
    direct_neigh_idxs = direct_neigh_idxs.astype(np.uint32)
    n_edges = n_edges.astype(np.uint32)
    mask = libgeo.unidirectional(uni_verts, direct_neigh_idxs, n_edges, target)
    mask = mask.astype(np.bool)
    #print(mask.shape, mask.dtype)
    c_source = np.array(source[mask], copy=True)
    c_target = np.array(target[mask], copy=True)
    c_distances = np.array(distances[mask], copy=True)
    return {
        "source": source,
        "target": target,
        "c_source": c_source,
        "c_target": c_target,
        "distances": distances,
        "c_distances": c_distances
    }


def get_geodesic_knns(target, distances, k, uni_verts=None, direct_neigh_idxs=None, n_edges=None, exclude_closest=True, source=None):
    if uni_verts is None or direct_neigh_idxs is None or n_edges is None:
        if source is None:
            raise Exception("source need to be inserted!")
        uni_verts, direct_neigh_idxs, n_edges = np.unique(source, return_index=True, return_counts=True)
    source_, target_, distances_, ok = libgeo.geodesic_knn(uni_verts, direct_neigh_idxs, n_edges, target, distances, k, exclude_closest)
    return {
        "source": source_,
        "target": target_,
        "distances": distances_,
    }, ok[0]


def apply_cp_(xyz, rgb, k_nn_adj=15, k_nn_geof=45, lambda_edge_weight=1, reg_strength=0.07, d_se_max=0, 
        mesh=False, uni_verts=None, direct_neigh_idxs=None, n_edges=None,
        source=None, target=None, distances=None, exclude_closest=True):
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    if mesh:
        if source is None or target is None or distances is None:
            raise Exception("Missing input arguments")
        graph_nn, ok = get_geodesic_knns(uni_verts=uni_verts, direct_neigh_idxs=direct_neigh_idxs, n_edges=n_edges,
            target=target, distances=distances, k=k_nn_adj, exclude_closest=exclude_closest)
        if not ok:
            print("Error while searching geodesic neighbourhood - apply euclidean KNN search")
            graph_nn, target_fea = graph_utils.compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof, verbose=False)
    else:
        graph_nn, target_fea = graph_utils.compute_graph_nn_2(xyz, k_nn_adj, k_nn_geof, verbose=False)
    graph_nn = unidirectional(
        graph_nn=graph_nn)
    geof = libply_c.compute_geof(xyz, graph_nn["target"], k_nn_adj, False).astype(np.float32)
    features = np.hstack((geof, rgb)).astype("float32") # add rgb as a feature for partitioning
    features[:,3] = 2. * features[:,3] # increase importance of verticality (heuristic)
    
    verbosity_level = 0.0
    speed = 2.0
    store_bin_labels = 0
    cutoff = 0 
    spatial = 0 
    weight_decay = 1
    graph_nn["edge_weight"] = np.array(1. / ( lambda_edge_weight + graph_nn["c_distances"] / np.mean(graph_nn["c_distances"])), dtype = "float32")
    point_idxs, p_vec, stats, duration = libcp.cutpursuit(features, graph_nn["c_source"], graph_nn["c_target"], 
        graph_nn["edge_weight"], reg_strength, cutoff, spatial, weight_decay, verbosity_level, speed, store_bin_labels)
    return point_idxs, p_vec, duration


def apply_cp(context, params):
    (k, reg_strength, use_edges) = params
    xyz, rgb, P, n_points, obj, colors_avaible = get_cloud(context=context)

    edges = obj.data.edges
    edgesList = []
    
    for i in range(len(edges)):
        edgesList.append([edges[i].vertices[0], edges[i].vertices[1]])
    edgesList = np.array(edgesList, dtype=np.uint32)
    
    reverseEdges = np.hstack((edgesList[:, 1][:, None], edgesList[:, 0][:, None]))
    edgesList = np.vstack((edgesList, reverseEdges))

    sortation = np.argsort(edgesList[:, 0])
    edgesList = edgesList[sortation, :]

    distances = np.sqrt(np.sum((xyz[edgesList[:, 0]] - xyz[edgesList[:, 1]])**2, axis=1))
    source = np.array(edgesList[:, 0], copy=True)
    target = np.array(edgesList[:, 1], copy=True)

    uni_verts, direct_neigh_idxs, n_edges = np.unique(source, return_index=True, return_counts=True)
    #print(uni_verts[0], direct_neigh_idxs[0], n_edges[0])
    source = source.astype(np.uint32)
    target = target.astype(np.uint32)
    uni_verts = uni_verts.astype(np.uint32)
    direct_neigh_idxs = direct_neigh_idxs.astype(np.uint32)
    n_edges = n_edges.astype(np.uint32)
    distances = distances.astype(np.float32)

    lambda_edge_weight = 1
    d_se_max = 0
    _, p_vec, _ = apply_cp_(xyz=xyz, rgb=rgb, k_nn_adj=k, k_nn_geof=k+3,
        lambda_edge_weight=lambda_edge_weight, reg_strength=reg_strength, d_se_max=d_se_max,
        mesh=use_edges, uni_verts=uni_verts, direct_neigh_idxs=direct_neigh_idxs, n_edges=n_edges,
        source=source, target=target, distances=distances)        
    create_meshes(p_vec=p_vec, xyz=xyz, rgb=rgb, obj=obj, algorithm="cut-pursuit")


def apply_vccs(context, params):
    (voxel_resolution, seed_resolution, color_importance, spatial_importance, normal_importance, refinement_iter,use_edges,r_search_gain) = params
    if use_edges:
        xyz, rgb, P, n_points, obj, colors_avaible, normals,\
        adjacency_list,\
        source, target, uni_verts, direct_neigh_idxs, n_edges, distances = get_graph(context=context)

        radius = r_search_gain * seed_resolution
        exclude_closest = False
        source_, target_, distances_, ok = libgeo.geodesic_radiusnn(uni_verts, direct_neigh_idxs, n_edges, target, distances, radius, exclude_closest)
        if not ok[0]:
            print("Error while searching geodesic neighbourhood - apply point cloud vccs")
            _, p_vec, _ = libvccs.vccs(P, voxel_resolution=voxel_resolution, seed_resolution=seed_resolution,
                color_importance=color_importance, spatial_importance=spatial_importance, normal_importance=normal_importance,
                refinementIter=refinement_iter)
        else:
            source_, target_, distances_, uni_verts_, direct_neigh_idxs_, n_edges_ = graph_utils.sort_graph(source=source_, target=target_, distances=distances_)
            source_ = source_.astype(np.uint32)
            target_ = target_.astype(np.uint32)
            distances_ = distances_.astype(np.float32)

            uni_verts_ = uni_verts_.astype(np.uint32)
            direct_neigh_idxs_ = direct_neigh_idxs_.astype(np.uint32)
            n_edges_ = n_edges_.astype(np.uint32)
            precalc = True
            point_idxs, p_vec, duration = libvccs.vccs_mesh(P, uni_verts_, direct_neigh_idxs_, n_edges_, source_, target_, distances_, 
                voxel_resolution=voxel_resolution, seed_resolution=seed_resolution, color_importance=color_importance,
                spatial_importance=spatial_importance, normal_importance=normal_importance, refinementIter=refinement_iter,
                r_search_gain=r_search_gain, precalc=precalc)
    else:
        xyz, rgb, P, n_points, obj, colors_avaible = get_cloud(context=context)
        _, p_vec, _ = libvccs.vccs(P, voxel_resolution=voxel_resolution, seed_resolution=seed_resolution,
            color_importance=color_importance, spatial_importance=spatial_importance, normal_importance=normal_importance,
            refinementIter=refinement_iter)
    create_meshes(p_vec=p_vec, xyz=xyz, rgb=rgb, obj=obj, algorithm="vccs")


def apply_vgs(context, params):
    (voxel_size,graph_size,sig_p,sig_n,sig_o,sig_e,sig_c,sig_w,cut_thred,points_min,adjacency_min,voxels_min,seed_size,sig_f,sig_a,sig_b,use_edges,r_search_gain) = params
    if use_edges:
        xyz, rgb, P, n_points, obj, colors_avaible, normals,\
        adjacency_list,\
        source, target, uni_verts, direct_neigh_idxs, n_edges, distances = get_graph(context=context)
                
        exclude_closest=False
        source_, target_, distances_, ok = libgeo.geodesic_radiusnn(uni_verts, direct_neigh_idxs, n_edges, target, distances, graph_size, exclude_closest)
        if not ok[0]:
            print("Error while searching geodesic neighbourhood - apply point cloud vgs")
            _, p_vec, _ = libvgs.vgs(P, voxel_size=voxel_size, graph_size=graph_size, sig_p=sig_p,
                sig_n=sig_n, sig_o=sig_o, sig_e=sig_e, sig_c=sig_c, sig_w=sig_w, cut_thred=cut_thred,
                points_min=points_min, adjacency_min=adjacency_min, voxels_min=voxels_min)
        else:
            source_, target_, distances_, uni_verts_, direct_neigh_idxs_, n_edges_ = graph_utils.sort_graph(source=source_, target=target_, distances=distances_)
            source_ = source_.astype(np.uint32)
            target_ = target_.astype(np.uint32)
            distances_ = distances_.astype(np.float32)

            uni_verts_ = uni_verts_.astype(np.uint32)
            direct_neigh_idxs_ = direct_neigh_idxs_.astype(np.uint32)
            n_edges_ = n_edges_.astype(np.uint32)
            _, p_vec, _ = libvgs.vgs_mesh(P, target_, direct_neigh_idxs_, n_edges_, distances_, normals, voxel_size=voxel_size, 
                sig_p=sig_p, sig_n=sig_n, sig_o=sig_o, sig_e=sig_e, sig_c=sig_c, sig_w=sig_w, cut_thred=cut_thred,
                points_min=points_min, adjacency_min=adjacency_min, voxels_min=voxels_min, use_normals=True)
    else:
        xyz, rgb, P, n_points, obj, colors_avaible = get_cloud(context=context)

        _, p_vec, _ = libvgs.vgs(P, voxel_size=voxel_size, graph_size=graph_size, sig_p=sig_p,
            sig_n=sig_n, sig_o=sig_o, sig_e=sig_e, sig_c=sig_c, sig_w=sig_w, cut_thred=cut_thred,
            points_min=points_min, adjacency_min=adjacency_min, voxels_min=voxels_min)
    create_meshes(p_vec=p_vec, xyz=xyz, rgb=rgb, obj=obj, algorithm="vgs")


def apply_svgs(context, params):
    (voxel_size,graph_size,sig_p,sig_n,sig_o,sig_e,sig_c,sig_w,cut_thred,points_min,adjacency_min,voxels_min,seed_size,sig_f,sig_a,sig_b,use_edges,r_search_gain) = params
    if use_edges:
        xyz, rgb, P, n_points, obj, colors_avaible, normals,\
        adjacency_list,\
        source, target, uni_verts, direct_neigh_idxs, n_edges, distances = get_graph(context=context)

        vccs_search_radius = r_search_gain * seed_size
        radius = max(graph_size, vccs_search_radius)
        exclude_closest=False
        source_, target_, distances_, ok = libgeo.geodesic_radiusnn(uni_verts, direct_neigh_idxs, n_edges, target, distances, radius, exclude_closest)
        if not ok[0]:
            print("Error while searching geodesic neighbourhood - apply point cloud vgs")
            _, p_vec, _ = libvgs.svgs(P, voxel_size=voxel_size, seed_size=seed_size, graph_size=graph_size, 
                sig_p=sig_p, sig_n=sig_n, sig_o=sig_o, sig_f=sig_f, sig_e=sig_e, sig_w=sig_w, 
                sig_a=sig_a, sig_b=sig_b, sig_c=sig_c, cut_thred=cut_thred,
                points_min=points_min, adjacency_min=adjacency_min, voxels_min=voxels_min)
        else:
            source_, target_, distances_, uni_verts_, direct_neigh_idxs_, n_edges_ = graph_utils.sort_graph(source=source_, target=target_, distances=distances_)
            source_ = source_.astype(np.uint32)
            target_ = target_.astype(np.uint32)
            uni_verts_ = uni_verts_.astype(np.uint32)
            direct_neigh_idxs_ = direct_neigh_idxs_.astype(np.uint32)
            n_edges_ = n_edges_.astype(np.uint32)
            distances_ = distances_.astype(np.float32)
            precalc=True

            # memory exception can occur if graph has too many edges!!!
            _, p_vec, _ = libvgs.svgs_mesh(P, source_, target_, uni_verts_, direct_neigh_idxs_, n_edges_, distances_, 
                voxel_size=voxel_size, seed_size=seed_size, graph_size=graph_size,
                sig_p=sig_p, sig_n=sig_n, sig_o=sig_o, sig_f=sig_f, sig_e=sig_e, sig_w=sig_w, 
                sig_a=sig_a, sig_b=sig_b, sig_c=sig_c, cut_thred=cut_thred,
                points_min=points_min, adjacency_min=adjacency_min, voxels_min=voxels_min, r_search_gain=r_search_gain, precalc=precalc)
    else:
        xyz, rgb, P, n_points, obj, colors_avaible = get_cloud(context=context)
        _, p_vec, _ = libvgs.svgs(P, voxel_size=voxel_size, seed_size=seed_size, graph_size=graph_size, 
            sig_p=sig_p, sig_n=sig_n, sig_o=sig_o, sig_f=sig_f, sig_e=sig_e, sig_w=sig_w, 
            sig_a=sig_a, sig_b=sig_b, sig_c=sig_c, cut_thred=cut_thred,
            points_min=points_min, adjacency_min=adjacency_min, voxels_min=voxels_min)
    create_meshes(p_vec=p_vec, xyz=xyz, rgb=rgb, obj=obj, algorithm="svgs")


def apply_ransac(context, params):
    (distance_threshold, num_iterations) = params
    xyz, rgb, P, n_points, obj, colors_avaible = get_cloud(context=context)

    plane = pyrsc.Plane()
    _, inliers = plane.fit(xyz, thresh=distance_threshold, maxIteration=num_iterations)

    inliers = np.array(inliers, dtype=np.uint32)
    remaining = np.arange(n_points, dtype=np.uint32)
    remaining = np.delete(remaining, inliers)
    point_idxs = [inliers, remaining]
    p_vec = -np.ones((xyz.shape[0], ), dtype=np.int32)
    for i in range(len(point_idxs)):
        idxs = point_idxs[i]
        p_vec[idxs] = i
    create_meshes(p_vec=p_vec, xyz=xyz, rgb=rgb, obj=obj, algorithm="plane")


def search_directed(np_edges_, a, b):
    # we search a directed edge (a,b)
    #print(np_edges_)
    # todo: vor der suche
    uni_edges, uni_idxs, uni_counts = np.unique(np_edges_[:, 0], return_index=True, return_counts=True)
    #print(uni_edges)
    idx = np.where(uni_edges == a)[0][0]
    #print(idx)
    start = uni_idxs[idx]
    stop = uni_counts[idx] + start
    #print(start, stop)
    edge_idx = np.where(np_edges_[start:stop, 1] == b)[0][0]
    #print(edge_idx)
    edge_idx += idx
    #print(edge_idx)
    return edge_idx


def sort_edges(np_edges):
    sortation = np.argsort(np_edges[:, 0])
    #print(sortation)
    np_edges_ = np_edges[sortation, :]
    return np_edges_


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def semantic_segmentation(context, gpu=0, batch_size=32, num_point=4096, log_dir="pointnet2_sem_seg", num_votes=1):
    '''
    PARAMETERS
    gpu (int):
        Specify the gpu device.
    batch_size (int):
        Batch size in testing [default: 32]
    num_point (int):
        number of input points [default: 4096]
    log_dir (str):
        Experiment root
    num_votes (int):
        Aggregate segmentation scores with voting [default: 5]
    '''
    classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair',
        'sofa', 'bookcase', 'board', 'clutter']
    class2label = {cls: i for i, cls in enumerate(classes)}
    seg_classes = class2label
    seg_label_to_cat = {}
    for i, cat in enumerate(seg_classes.keys()):
        seg_label_to_cat[i] = cat

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    NUM_CLASSES = 13
    BATCH_SIZE = batch_size
    NUM_POINT = num_point
    plugin_dir = 'D:/Projects/CalvinsBlenderAddOn'
    experiment_dir = plugin_dir + '/log/sem_seg/' + log_dir

    # load the model
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    classifier = pointnet2_sem_seg.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
    print("Apply the semantic segmentation with ", model_name)

    num_point, block_size, sample_rate = 4096, 1.0, 0.01
    

    xyz, rgb, _, _, obj, colors_avaible = get_cloud(context=context)
    divide = False
    if np.max(rgb) <= 2:
        rgb *= 255
        divide = True
    #print("Range of rgb colors: [{0}, {1}]".format(np.min(rgb), np.max(rgb)))
    labels = np.ones((xyz.shape[0], 1))
    P = np.hstack((xyz, rgb, labels))

    TEST_DATASET_WHOLE_SCENE = CustomScene(P=P, block_points=NUM_POINT)
    print("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    with torch.no_grad():
        num_batches = len(TEST_DATASET_WHOLE_SCENE)
        assert num_batches == 1
        #for batch_idx in range(num_batches):
        batch_idx = 0
        whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
        whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
        vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
        for _ in range(num_votes):
            scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
            num_blocks = scene_data.shape[0]
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
            batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

            batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
            batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

            for sbatch in range(s_batch_num):
                start_idx = sbatch * BATCH_SIZE
                end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                real_batch_size = end_idx - start_idx
                batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                batch_data[:, :, 3:6] /= 1.0

                torch_data = torch.Tensor(batch_data)
                torch_data = torch_data.float().cuda()
                torch_data = torch_data.transpose(2, 1)
                seg_pred, _ = classifier(torch_data)
                batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                           batch_pred_label[0:real_batch_size, ...],
                                           batch_smpw[0:real_batch_size, ...])

        pred_label = np.argmax(vote_label_pool, 1)
        if divide:
            rgb /= 255
        create_meshes(p_vec=pred_label, xyz=xyz, rgb=rgb, obj=obj, classes=classes, algorithm="pointnet++")