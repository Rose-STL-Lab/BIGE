"""
Code inspired from: https://skel.is.tue.mpg.de 

Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See https://skel.is.tue.mpg.de/license.html for licensing and contact information.
"""

import os
import shutil
import numpy as np
import tqdm
import trimesh

import nimblephysics as nimble
import pickle as pkl

import sys
import inspect
print(os.path.dirname(os.path.abspath(os.path.__file__)))


if "visualizer" in os.getcwd(): 
    sys.path.insert(0, os.path.dirname(os.getcwd()))
else: 
    sys.path.insert(0, os.getcwd())



from utils.motion_process import recover_from_ric


def load_osim(osim_path, geometry_path, ignore_geometry=False):
    """Load an osim file"""
       
    assert os.path.exists(osim_path), f'Could not find osim file {osim_path}'
    osim_path = os.path.abspath(osim_path)

    if geometry_path is not None: 
        # Check that there is a Geometry folder at the same level as the osim file
        geometry_path = os.path.join(args.out_dir, 'OpenCap_LaiArnoldModified2017_Geometry') 
    
    if os.path.dirname(osim_path) != os.path.dirname(geometry_path):        
        # Check that there is a Geometry folder at the same level as the osim file. Otherwise nimble physics cannot import it. 
        os.makedirs(os.path.join(os.path.dirname(osim_path),'Geometry'), exist_ok=True)
        for file in os.listdir(geometry_path):
            if os.path.exists(os.path.join(os.path.dirname(osim_path),'Geometry',file)): continue
            os.symlink(os.path.join(geometry_path, file), os.path.join(os.path.dirname(osim_path), 'Geometry', file))
    

        


    # Create a tmp folder
    osim : nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(osim_path)



    assert osim is not None, "Could not load osim file: {}".format(osim_path)
    return osim

class OSIMSequence():
    """
    Represents a temporal sequence of OSSO poses. Can be loaded from disk or initialized from memory.
    """

    def __init__(self,
                 osim,
                 motion,
                 color_markers_per_part = False,
                 color_markers_per_index = False, # Overrides color_markers_per_part
                 color_skeleton_per_part = False,
                 osim_path = None,
                 fps = None,
                 fps_in = None,
                 is_rigged = False,
                 viewer = True,
                 **kwargs):
        """
        Initializer.
        :param osim_path: A osim model
        :param mot: A motion array 
        :osim_path: Path the osim model was loaded from (optional)
        :param kwargs: Remaining arguments for rendering.
        """
        self.osim_path = osim_path
        self.osim = osim
        self.motion = motion

        assert self.osim_path, "No osim path given"

        self.fps = fps
        self.fps_in = fps_in

        self._is_rigged = is_rigged or True

        assert len(motion.shape) == 2

        self.n_frames = motion.shape[0]

        self._render_kwargs = kwargs

        # The node names of the skeleton model, the associated mesh and the template indices
        self.node_names = [n.getName() for n in osim.skeleton.getBodyNodes()]
        
        self.meshes_dict = {}
        self.indices_dict = {}
        self.generate_meshes_dict() # Populate self.meshes_dict and self.indices_dict
        self.create_template()

        # model markers
        markers_labels = [ml for ml in self.osim.markersMap.keys()]
        markers_labels.sort()
        self.markers_labels = markers_labels

        # Nodes
        self.vertices, self.faces, self.marker_trajectory, self.joints, self.joints_ori = self.fk()
        
        # TODO: fix that. This triggers a segfault at destruction so I hardcode it
        # self.joints_labels = [J.getName() for J in self.osim.skeleton.getJoints()]
        # self.joints_labels = ['ground_pelvis', 'hip_r', 'walker_knee_r', 'ankle_r', 'subtalar_r', 'mtp_r', 'hip_l', 'walker_knee_l', 'ankle_l', 'subtalar_l', 'mtp_l', 'back', 'neck', 'acromial_r', 'elbow_r', 'radioulnar_r', 'radius_hand_r', 'acromial_l', 'elbow_l', 'radioulnar_l', 'radius_hand_l']
    

    def per_part_bone_colors(self):
        """ Color the mesh with one color per node. """
        vertex_colors = np.ones((self.n_frames, self.template.vertices.shape[0], 4))
        color_palette = vertex_colors_from_weights(np.arange(len(self.node_names)), shuffle=True)
        for i, node_name in enumerate(self.node_names):
            id_start, id_end = self.indices_dict[node_name]
            vertex_colors[:, id_start :id_end, 0:3] = color_palette[i, :]
        return vertex_colors


    def generate_meshes_dict(self):
        """ Output a dictionary giving for each bone, the attached mesh"""

        current_index = 0
        self.indices_dict = {}
        self.meshes_dict = {}

        node_names = self.node_names
        for node_name in node_names:
            mesh_list = []
            body_node = self.osim.skeleton.getBodyNode(node_name)
            # print(f' Loading meshes for node: {node_name}')
            num_shape_nodes = body_node.getNumShapeNodes()
            if num_shape_nodes == 0:
                print(f'WARNING:\tNo shape nodes listed for  {node_name}')
            for shape_node_i in range(num_shape_nodes):
                shape_node = body_node.getShapeNode(shape_node_i)
                submesh_path = shape_node.getShape().getMeshPath()
                # Get the scaling for this meshes
                scale = shape_node.getShape().getScale()
                offset = shape_node.getRelativeTranslation()
                # Load the mesh
                try:
                    submesh = trimesh.load_mesh(submesh_path, process=False)
                    # print(f'Loaded mesh {submesh_path}')
                except Exception as e:
                    print(e)
                    print(f'WARNING:\tCould not load mesh {submesh_path}')
                    submesh = None
                    continue
                
                if submesh is not None:
                    trimesh.repair.fix_normals(submesh)
                    trimesh.repair.fix_inversion(submesh)
                    trimesh.repair.fix_winding(submesh)

                    # import pyvista
                    # submesh_poly = pyvista.read(submesh_path)
                    # faces_as_array = submesh_poly.faces.reshape((submesh_poly.n_faces, 4))[:, 1:] 
                    # submesh = trimesh.Trimesh(submesh_poly.points, faces_as_array) 

                    # Scale the bone to match .osim subject scaling
                    submesh.vertices[:] = submesh.vertices * scale
                    submesh.vertices[:] += offset
                    # print(f'submesh_path: {submesh_path}, Nb vertices: {submesh.vertices.shape[0]}')
                    mesh_list.append(submesh)

            # Concatenate meshes
            if mesh_list:
                node_mesh = trimesh.util.concatenate(mesh_list)
                self.indices_dict[node_name] = (current_index, current_index + node_mesh.vertices.shape[0])
                current_index += node_mesh.vertices.shape[0]
            else:
                node_mesh = None
                print("\t WARNING: No submesh for node:", node_name)
                self.indices_dict[node_name] = (current_index, current_index )
            
            # Add to the dictionary
            self.meshes_dict[node_name] = node_mesh
        print(self.meshes_dict)


    def create_template(self):

        part_meshes = []
        for node_name in self.node_names:
            mesh = self.meshes_dict[node_name]
            # assert mesh, "No mesh for node: {}".format(node_name)
            if mesh is None:
                print( "WARNING: No mesh for node: {}".format(node_name))
            if mesh:
                part_meshes.append(mesh)
        # part_meshes = [m for m in part_meshes if m]
        template = trimesh.util.concatenate(part_meshes)
        # import ipdb; ipdb.set_trace()
        
        template.remove_degenerate_faces()
        self.template = template

        #save mesh
        # # import ipdb; ipdb.set_trace()
        # self.template.export('template.obj')
        # print(f'Saved template to template.obj')

        # from psbody.mesh import Mesh
        # m = Mesh(filename='template.obj')
        # m.set_vertex_colors_from_weights(np.arange(m.v.shape[0]))
        # m.show()


    @classmethod
    def a_pose(cls, osim_path = None, **kwargs):
        """Creates a OSIM sequence whose single frame is a OSIM mesh in rest pose."""
        # Load osim file
        if osim_path is None:
            osim : nimble.biomechanics.OpenSimFile = nimble.models.RajagopalHumanBodyModel()
            osim_path = "RajagopalHumanBodyModel.osim" # This is not a real path, but it is needed to instantiate the sequence object
        else:
            osim = load_osim(osim_path)
            
        assert osim is not None, "Could not load osim file: {}".format(osim_path)
        motion = osim.skeleton.getPositions()[np.newaxis,:]

        return cls(osim, motion,
                    osim_path = osim_path,
                    **kwargs)
        
    @classmethod
    def zero_pose(cls, osim_path = None, **kwargs):
        """Creates a OSIM sequence whose single frame is a OSIM mesh in rest pose."""
        # Load osim file
        if osim_path is None:
            osim : nimble.biomechanics.OpenSimFile = nimble.models.RajagopalHumanBodyModel()
            osim_path = "RajagopalHumanBodyModel.osim" # This is not a real path, but it is needed to instantiate the sequence object
        else:
            osim = nimble.biomechanics.OpenSimParser.parseOsim(osim_path)
            
        assert osim is not None, "Could not load osim file: {}".format(osim_path)

        # motion = np.zeros((1, len(osim.skeleton.getBodyNodes())))
        motion = osim.skeleton.getPositions()[np.newaxis,:]
        motion = np.zeros_like(motion)
        # import ipdb; ipdb.set_trace()

        return cls(osim, motion,
                    osim_path = osim_path,
                    **kwargs)


    @classmethod
    def from_ab_folder(cls, ab_folder, trial, start_frame=None, end_frame=None, fps_out=None, **kwargs):   
        """
        Load an osim sequence from a folder returned by AddBiomechanics
        ab_folder: the folder returned by AddBiomechanics, ex: '/home/kellerm/Data/AddBiomechanics/CMU/01/smpl_head_manual'
        trial: Trial name
        start_frame: the first frame to load
        end_frame: the last frame to load
        fps_out: the output fps
        """
        
        if ab_folder[-1] != '/':
            ab_folder += '/'

        mot_file = ab_folder + f"IK/{trial}_ik.mot"
        osim_path = ab_folder + 'Models/optimized_scale_and_markers.osim'

        
        return OSIMSequence.from_files(osim_path=osim_path, mot_file=mot_file, start_frame=start_frame, end_frame=end_frame, fps_out=fps_out, **kwargs)



    @classmethod
    def from_files(cls, osim_path, mot_file, geometry_path=None, start_frame=None, end_frame=None, fps_out: int=None, ignore_fps=False, ignore_geometry=False,**kwargs):
        """Creates a OSIM sequence from addbiomechanics return data
        osim_path: .osim file path
        mot_file : .mot file path
        start_frame: first frame to use in the sequence
        end_frame: last frame to use in the sequence
        fps_out: frames per second of the output sequence
        ignore_geometry : use the aitconfig.osim_geometry folder instead of the one next to the osim file
        """

        # Load osim file
        osim = load_osim(osim_path, geometry_path=geometry_path, ignore_geometry=ignore_geometry)

        # Load the .mot file
        mot: nimble.biomechanics.OpenSimMot = nimble.biomechanics.OpenSimParser.loadMot(
                    osim.skeleton, mot_file)

        motion = np.array(mot.poses.T)    

        # Crop and sample
        sf = start_frame or 0
        ef = end_frame or motion.shape[0]
        motion = motion[sf:ef]

        # estimate fps_in
        ts = np.array(mot.timestamps)   
        fps_estimated = 1/np.mean(ts[1:] - ts[:-1])
        fps_in = int(round(fps_estimated)) 
        print(f'Estimated fps for the .mot sequence: {fps_estimated}, rounded to {fps_in}')
        
        if not ignore_fps:
            assert abs(1 - fps_estimated/fps_in) < 1e-5 , f"FPS estimation might be bad, {fps_estimated} rounded to {fps_in}, check."

            if fps_out is not None:
                assert fps_in%fps_out == 0, 'fps_out must be a interger divisor of fps_in'
                mask = np.arange(0, motion.shape[0], fps_in//fps_out)
                print(f'Resampling from {fps_in} to {fps_out} fps. Keeping every {fps_in//fps_out}th frame')
                # motion = resample_positions(motion, fps_in, fps_out) #TODO: restore this 
                motion = motion[mask]   
        
                
            del mot
        else:
            fps_out = fps_in

        return cls(osim, motion, osim_path=osim_path, fps=fps_out, fps_in=fps_in, **kwargs)

    @staticmethod
    def to_numpy(x):
        if isinstance(x, np.ndarray):
            return x
        return x.detach().cpu().numpy()

    def fk(self):
        """Get vertices from the poses."""
        # Forward kinematics https://github.com/nimblephysics/nimblephysics/search?q=setPositions

        verts = np.zeros((self.n_frames, self.template.vertices.shape[0], self.template.vertices.shape[1]))
        markers = np.zeros((self.n_frames, len(self.markers_labels), 3))

        joints = np.zeros([self.n_frames, len(self.meshes_dict), 3])
        joints_ori = np.zeros([self.n_frames, len(self.meshes_dict), 3, 3])

        prev_verts = verts[0]
        prev_pose = self.motion[0, :]
        
        for frame_id in (pbar := tqdm.tqdm(range(self.n_frames))):
            pbar.set_description("Generating osim skeleton meshes ")

            pose = self.motion[frame_id, :]
            # If the pose did not change, use the previous frame verts
            if np.all(pose == prev_pose) and frame_id != 0:
                verts[frame_id] = prev_verts
                continue

            # Pose osim
            self.osim.skeleton.setPositions(self.motion[frame_id, :])

            # Since python 3.6, dicts have a fixed order so the order of this list should be marching labels
            markers[frame_id, :, :] = np.vstack(self.osim.skeleton.getMarkerMapWorldPositions(self.osim.markersMap).values())
            #Sanity check for previous comment
            assert list(self.osim.skeleton.getMarkerMapWorldPositions(self.osim.markersMap).keys()) == self.markers_labels, "Marker labels are not in the same order"

            for ni, node_name in enumerate(self.node_names):
                # if ('thorax' in node_name) or ('lumbar' in node_name):
                #     # We do not display the spine as the riggidly rigged mesh can't represent the constant curvature of the spine
                #     continue
                mesh = self.meshes_dict[node_name]
                if mesh is not None:

                    part_verts = mesh.vertices

                    # pose part
                    transfo = self.osim.skeleton.getBodyNode(node_name).getWorldTransform()
                    
                    # Add a row of homogenous coordinates 
                    part_verts = np.concatenate([part_verts, np.ones((mesh.vertices.shape[0], 1))], axis=1)
                    part_verts = np.matmul(part_verts, transfo.matrix().T)[:,0:3]
                        
                    # Update the part in the full mesh       
                    id_start, id_end = self.indices_dict[node_name]
                    verts[frame_id, id_start :id_end, :] = part_verts

                    # Update joint                    
                    joints[frame_id, ni, :] = transfo.translation()
                    joints_ori[frame_id, ni, :, :] = transfo.rotation()
            

            prev_verts = verts[frame_id]
            prev_pose = pose

            
        faces = self.template.faces

        return self.to_numpy(verts), self.to_numpy(faces), markers, joints, joints_ori
        

import torch 
import polyscope as ps
import polyscope.imgui as psim
class OSIMRetargetter: 
    def __init__(self,exp_dir='./output-viz/'):

        self.osim = OSIMSequence.a_pose()

        self.target_filepath = None
        self.target_joints = None 
        self.minT = 20
        self.T = 130 # Default value
        self.t = 0


        # All data
        self.exp_dir = exp_dir
        self.exps = [file for file in os.listdir(self.exp_dir) if file.endswith('.npy')]

        # Categories 
        from classifiers import desc_to_action
        
        self.categories = [ x.replace('full', 'fast') for x in  desc_to_action]

        self.polyscope_scene = {
            "is_true1": False,
            "is_true2": True,
            "ui_int": 7,
            "ui_float1": -3.2,
            "ui_float2": 0.8,
            "ui_color3": (1., 0.5, 0.5),
            "ui_color4": (0.3, 0.5, 0.5, 0.8),
            "ui_angle_rad": 0.2,
            "ui_text": "Enter instructions here",
            "experiment_options": self.exps,
            "experiment_options_selected": self.exps[0],

            "category_options": self.categories,
            "category_options_selected": self.categories[0],

            "index": 1,

            "is_paused": False
        }


        # bodyJoints = [skeleton.getBodyNode(i).getName() for i in range(skeleton.getNumBodyNodes())]
        # ['ground_pelvis', 'hip_r', 'walker_knee_r', 'ankle_r', 'subtalar_r', 'mtp_r', 'hip_l', 'walker_knee_l', 'ankle_l', 'subtalar_l', 'mtp_l', 'back', 'acromial_r', 'elbow_r', 'radioulnar_r', 'radius_hand_r', 'acromial_l', 'elbow_l', 'radioulnar_l', 'radius_hand_l']
        # smpl_joints = [
        # 'pelvis',    'left_hip',    'right_hip',    'spine1',    'left_knee',    'right_knee',
        # 'spine2',    'left_ankle',  'right_ankle',  'spine3',    'left_foot',    'right_foot', 
        # 'neck',    'left_collar',   'right_collar', 'head',      'left_shoulder','right_shoulder',
        # 'left_elbow','right_elbow', 'left_wrist',   'right_wrist',    'left_hand',    'right_hand']

        self.mapping_bodyJoints = {
            # 'back': 3,  # Pelvis
            'hip_l': 1,  # Left hip
            'hip_r': 2,  # Right hip
            'walker_knee_l': 4,  # Left knee
            'walker_knee_r': 5,  # Right knee
            'ankle_l': 7,  # Left ankle
            'ankle_r': 8,  # Right ankle
            'mtp_l': 10,  # Left mtp
            'mtp_r': 11,  # Right mtp

            'acromial_l': 16,  # Left Shoulder
            'acromial_r': 17,  # Right shoulder
            'elbow_l': 18,  # Left elbow
            'elbow_r': 19,  # Right elbow

            # Remove for some excercises
            'radius_hand_l': 20,  # Left wrist
            'radius_hand_r': 21,  # Right wrist
            }

        # osim_dict = dict([(joint.getName(),i) for i,joint in enumerate(self.osim.osim.skeleton.getJoints())]) # Causes segfault for some reason when calling skeleton.getPoisons after this. Could be related to: https://github.com/keenon/nimblephysics/issues/184   		
        osim_dict = ['ground_pelvis', 'hip_r', 'walker_knee_r', 'ankle_r', 'subtalar_r', 'mtp_r', 'hip_l', 'walker_knee_l', 'ankle_l', 'subtalar_l', 'mtp_l', 'back', 'acromial_r', 'elbow_r', 'radioulnar_r', 'radius_hand_r', 'acromial_l', 'elbow_l', 'radioulnar_l', 'radius_hand_l']
        osim_dict = dict([(name,i) for i,name in enumerate(osim_dict)])
        self.osim_index = np.array([osim_dict[name] for name in self.mapping_bodyJoints])
        self.smpl_index = np.array([self.mapping_bodyJoints[name] for name in self.mapping_bodyJoints])


    def load_joints(self,motion_path,scale=1.0):
        motions = np.load(os.path.join(motion_path))
        num_joints = 22
        motions = recover_from_ric(torch.from_numpy(motions).float().cuda(), num_joints)
        motions = motions.detach().cpu().numpy()
        motions[:,:,2] *= -1 # Replace z-axis with -z-axis.
        print(f'completed loading {motion_path} with shape: {motions.shape}') 

        motions_mean = np.mean(motions, axis=(0,1),keepdims=True)
        motions = scale*(motions - motions_mean) + motions_mean
  
        self.target_joints = motions
        self.target_filepath = motion_path
        self.T = motions.shape[0]

    

    def retarget(self,lambda_temporal=0.1,max_epochs=1): 
        
        return  
    
        skeleton = self.osim.osim.skeleton

        dof = skeleton.getPositions().shape[0]
        mot_data = np.zeros((self.T, dof)) 
        mot_data[:,2] = -np.pi/2


        bodyJoints = [skeleton.getJoint(name) for name in self.mapping_bodyJoints]
        target_joints_indices = [self.mapping_bodyJoints[name] for name in self.mapping_bodyJoints]
    
        # Repeat wrist indices 
        bodyJoints += bodyJoints[-2:]
        target_joints_indices += target_joints_indices[-2:]

        target_joints_indices = np.array(target_joints_indices)

        best_error_timestep = [np.inf]*self.T 
        for t in range(self.T-1,-1,-1):
            # mot_data[1:-1] = (1-lambda_temporal) * mot_data[1:-1] + lambda_temporal//2 * mot_data[0:-2] + lambda_temporal//2 * mot_data[2:]
            # for epochs in range(3):
            target_joints = self.target_joints[t,target_joints_indices].astype(np.float64).reshape((-1,1))
            skeleton.setPositions(mot_data[t])
            for i in range(max_epochs):
                err = skeleton.fitJointsToWorldPositions(bodyJoints, target_joints, scaleBodies=True,logOutput=False,lineSearch=True)
                if np.abs(err - best_error_timestep[t]) < 1e-4:
                    break 
                
                best_error_timestep[t] = err
                mot_data[t] = skeleton.getPositions()

        # target_joints = self.target_joints[best_error_timestep,target_joints_indices].astype(np.float64).reshape((-1,1))    
        # err = skeleton.fitJointsToWorldPositions(bodyJoints, target_joints, scaleBodies=True,logOutput=True,lineSearch=True)

        # avg_error = 0
        # for t in range(self.T-1,-1,-1):
        #     target_joints = self.target_joints[t,target_joints_indices].astype(np.float64).reshape((-1,1))
            
        #     err = skeleton.fitJointsToWorldPositions(bodyJoints, target_joints, scaleBodies=False,logOutput=True,lineSearch=True)
        #     mot_data[t] = skeleton.getPositions()

        #     if err < best_error:
        #         best_error = err
        #         best_error_timestep = t

        self.osim = OSIMSequence.a_pose()

        self.osim.motion = mot_data
        self.osim.n_frames = mot_data.shape[0]
        self.osim.vertices, self.osim.faces, self.osim.marker_trajectory, self.osim.joints, self.osim.joints_ori = self.osim.fk()

        self.best_error_timestep = best_error_timestep




    def save(self,save_path):
        # Save .mot file
        
        assert self.mot_data.shape[0] == self.T, "Invalid mot_data shape: {} != {}".format(self.mot_data.shape[0], self.T)

        dof_names =['time'] + [skeleton.getDofByIndex(i).getName() for i in range(skeleton.getNumDofs())]
        headers = ["Coordinates","version=1",f"nRows={self.T}", f"nColumns={len(dof_names)}", 
            "inDegrees=yes", # Not sure about this
            "Units are S.I. units (second, meters, Newtons, ...)",
            "If the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).",
            "",
            "endheader"
            ]
        
        with open(save_path, 'w') as f:
            f.write('\n'.join(headers))
            f.write(' '.join(dof_names) + '\n')
            for t in range(self.T):
                f.write(' '.join([str(x) for x in mot_data[t]]) + '\n')      
        



    def render(self):
        ps.init()
        ps.set_ground_plane_mode('tile_reflection')


        bone_array = [0,0, 0, 0,1, 2, 3, 4, 5, 6, 7,8,9,9,9,12,13,14,16,17,18,19,20,21]
        smpl_bone_array = np.array([[i,p] for i,p in enumerate(bone_array)])

        self.smpl_skeleton = ps.register_curve_network("My skelton", self.target_joints[0], smpl_bone_array[:22])

        # self.ps_biomechnical = ps.register_surface_mesh("Biomechnical Model",self.osim.vertices[0],self.osim.faces,transparency=0.7,color=np.array([1,1,1]))
        # self.ps_biomechnical_joints = ps.register_point_cloud("Biomechnical Joints",self.osim.joints[0],color=np.array([0,0,0]))

        joint_mapping = np.concatenate([self.target_joints[0,self.smpl_index],self.osim.joints[0,self.osim_index]],axis=0)
        joint_mapping_edges = np.array([(i,joint_mapping.shape[0]//2+i) for i in range(joint_mapping.shape[0]//2)])
        
        # self.ps_joint_mapping = ps.register_curve_network(f"Mapping (target- smpl) joints",joint_mapping,joint_mapping_edges,radius=0.001,color=np.array([0,1,0]))

        # ps.set_ground_plane_height_factor(0)

        ps.set_user_callback(self.callback)
        
        ps.show()

    

    def callback(self):
        
        if self.t < self.minT: 
            self.t = self.minT

        ########### Checks ############
        # Ensure self.t lies between 
        self.t %= self.T

        ### Update animation based on self.t
        if hasattr(self, 'smpl_skeleton'):
            self.smpl_skeleton.update_node_positions(self.target_joints[self.t])
        
        if hasattr(self, 'ps_biomechnical'):
            self.ps_biomechnical.update_vertex_positions(self.osim.vertices[self.t])
        
        if hasattr(self, 'ps_biomechnical_joints'):
            self.ps_biomechnical_joints.update_point_positions(self.osim.joints[self.t])
        
        if hasattr(self, 'ps_joint_mapping'):
            joint_mapping = np.concatenate([self.target_joints[self.t,self.smpl_index],self.osim.joints[self.t,self.osim_index]],axis=0)
            self.ps_joint_mapping.update_node_positions(joint_mapping)


        if not self.polyscope_scene['is_paused']: 
            self.t += 1 


        # Check keyboards for inputs
        
        # Check for spacebar press to toggle pause
        if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_Space)) or psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_Space)):
            
            self.polyscope_scene['is_paused'] = not self.polyscope_scene['is_paused']

        # Left arrow pressed
        if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_LeftArrow)) or psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_LeftArrow)):
            self.t -= 1

        if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_RightArrow)) or psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_RightArrow)):
            self.t += 1

        # Input text
        changed, self.polyscope_scene["ui_text"] = psim.InputText("- Coach Instructions", self.polyscope_scene["ui_text"])


        ############## Create the GUI to update the animations 
        # psim.Begin("Video Controller",True)


        # psim.SetWindowPos((1340,100.0),1) # Set the position the window at the bottom of the GUI
        # psim.SetWindowSize((500.0,700.0),1)

        # Create a floater to show the timestep and adject self.t accordingly
        changed, self.t = psim.SliderInt("", self.t, v_min=self.minT, v_max=self.T)
        psim.SameLine()

        # Create a render button which when pressed will create a .mp4 file
        if psim.Button("<"):
            self.t -= 1
        
        psim.SameLine()
        if psim.Button("Play Video" if self.polyscope_scene['is_paused'] else "Pause Video"):
            self.polyscope_scene['is_paused'] = not self.polyscope_scene['is_paused']

        psim.SameLine()
        if psim.Button(">"):
            self.t += 1

        # psim.SameLine()
        if psim.Button("Render Video"):
            self.render_video()        

        if(psim.TreeNode("Load Experiment")):

            # psim.TextUnformatted("Load Optimized samples")

            changed_exp = psim.BeginCombo("- Experiement", self.polyscope_scene["experiment_options_selected"])
            if changed_exp:
                for val in self.polyscope_scene["experiment_options"]:
                    _, selected = psim.Selectable(val, selected=self.polyscope_scene["experiment_options_selected"]==val)
                    if selected:
                        self.polyscope_scene["experiment_options_selected"] = val
                psim.EndCombo()
            
            if(psim.Button("Load Optimized samples")):
                filepath = os.path.join(self.exp_dir,self.polyscope_scene['experiment_options_selected'])
                self.load_joints(filepath)
                self.retarget()
                # ps.set_ground_plane_height_factor(np.min(self.osim.joints[:,:,1]))
            psim.TreePop()


        # psim.End()

    def render_video(self):
                
        os.makedirs('/tmp/skeleton/',exist_ok=True)
        for t in range(self.T):
            ### Update animation based on self.t
            if hasattr(self, 'smpl_skeleton'):
                self.smpl_skeleton.update_node_positions(self.target_joints[t])
            
            if hasattr(self, 'ps_biomechnical'):
                self.ps_biomechnical.update_vertex_positions(self.osim.vertices[t])
            
            if hasattr(self, 'ps_biomechnical_joints'):
                self.ps_biomechnical_joints.update_point_positions(self.osim.joints[t])
            
            if hasattr(self, 'ps_joint_mapping'):
                joint_mapping = np.concatenate([self.target_joints[t,self.smpl_index],self.osim.joints[t,self.osim_index]],axis=0)
                self.ps_joint_mapping.update_node_positions(joint_mapping)

            ps.screenshot(f"/tmp/skeleton/{t}.png",transparent_bg=False)

        os.system(f"ffmpeg  -y -i /tmp/skeleton/%d.png -pix_fmt yuv420p {self.target_filepath.replace('.npy', '.mp4')}")


if __name__ == "__main__": 
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=None, help='Location where predictions are stored.')
    parser.add_argument('--motion-list', default=None, nargs="+", type=str, help="motion name list")
    parser.add_argument("--file", type=str, default=None, help='motion npy file')

    args = parser.parse_args()

    assert args.file is not None or \
        (args.motion_list is not None and args.out_dir is not None) or \
        args.out_dir is not None,\
        "Please provide --file or  --motion-list or --out-dir"

    if args.file is not None:
        osim_retargetter = OSIMRetargetter(exp_dir=os.path.dirname(args.file))
        osim_retargetter.load_joints(args.file)

        osim_retargetter.retarget()
        osim_retargetter.render()
        osim_retargetter.save(save_path=args.file.replace('.npy', '.mot'))

    elif args.motion_list is not None and args.out_dir is not None:
        osim_retargetter = OSIMRetargetter(exp_dir=args.out_dir)
        if args.out_dir is not None:
            args.motion_list = [os.path.join(args.out_dir, x) for x in args.motion_list]    
        
        for filename in args.motion_list:
            osim_retargetter.load_joints(filename)
            osim_retargetter.retarget()
            osim_retargetter.save(save_path=filename.replace('.npy', '.mot'))

    elif args.out_dir is not None:
        osim_retargetter = OSIMRetargetter(exp_dir=args.out_dir)
        for filename in os.listdir(args.out_dir):
            if filename.endswith('.npy'):
                osim_retargetter.load_joints(os.path.join(args.out_dir, filename))
                osim_retargetter.retarget()
                osim_retargetter.save(save_path=os.path.join(args.out_dir, filename.replace('.npy', '.mot')))

    verts, faces, markers, joints, joints_ori = osim_retargetter.fk()
    print(verts.shape, faces.shape, markers.shape, joints.shape, joints_ori.shape)