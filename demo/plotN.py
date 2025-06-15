import os 
import sys
import json
import numpy as np 
from tqdm import tqdm
import polyscope as ps 
import polyscope.imgui as psim
import matplotlib.pyplot as plt


file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
UCSD_OpenCap_Fitness_Dataset_path = os.path.join(dir_path,'..', '..', 'UCSD-OpenCap-Fitness-Dataset' , 'src')
UCSD_OpenCap_Fitness_Dataset_path = os.path.abspath(UCSD_OpenCap_Fitness_Dataset_path)
sys.path.append(UCSD_OpenCap_Fitness_Dataset_path)
print(UCSD_OpenCap_Fitness_Dataset_path)



from utils import * 
from dataloader import OpenCapDataLoader,MultiviewRGB
from smpl_loader import SMPLRetarget

# from osim import OSIMSequence
# Load LaiArnoldModified2017
from osim import OSIMSequence

from scipy.spatial.transform import Rotation as sRotation
from scipy.interpolate import CubicSpline

def time_normalization(time_series,duration=101): 

	orig_time_space = np.linspace(0,1,len(time_series))
		
	spline = CubicSpline(orig_time_space, time_series)

	spline_input = np.linspace(0,1,duration)
	split_output = spline(spline_input)
			
	return split_output

class Visualizer: 
	def __init__(self): 
		
		ps.init()

		ps.remove_all_structures()
		# Set camera 
		ps.set_automatically_compute_scene_extents(True)
		ps.set_navigation_style("free")
		ps.set_view_projection_mode("orthographic")
		ps.set_ground_plane_mode('tile_reflection')

	def read_display_size(self):
		polyscope_ini_path = os.path.join(os.getcwd(),'.polyscope.ini')
		if not os.path.exists(polyscope_ini_path):
			self.display_size = (1280,720)
			ps.warning("Polyscope.ini not found, defaulting to 1280x720")
		else:
			with open(polyscope_ini_path) as f: 
				display_data = json.load(f)

			self.display_size = (display_data['windowWidth'], display_data['windowHeight'])

		
		return self.display_size

	def get_reds_color(self,vals,min_val=0,max_val=1):

		vals = vals.astype(np.float32)
		vals = (vals -min_val)/(max_val - min_val) # Normalize 

		colors = plt.get_cmap('viridis')(vals)

		return colors[...,:3]

	@staticmethod
	def load_mot(file): 
		with open(file,'r') as f:
			file_data = f.read().split('\n')
			# print(file_data)
			data = {'info':'', 'poses': []}
			read_header = False
			read_rows = 0
			
			for line in file_data:
				line = line.strip()
				if len(line) == 0:
					continue
				
				if not read_header:
					if line == 'endheader':
						read_header = True
						continue
					if '=' not in line:
						data['info'] += line + '\n'
					else:
						k,v = line.split('=')
						if v.isnumeric():
							data[k] = int(v)
						else:
							data[k] = v
				else:
					rows = line.split()
					if read_rows == 0:
						data['headers'] = rows
					else:
						rows = [float(row) for row in rows]
						data['poses'].append(rows)

					read_rows += 1
			data['headers'] = data['headers'][-80:]
			data['activations'] = np.array(data['poses'])[:,-80:] # Change to remove time 
			# data['poses'] = np.array(data['poses'])[:,1:34] # Change to remove time 
			return data



	def callback(self):
		
		########### Checks ############
		# Ensure self.t lies between 
		self.ps_data['t'] %= self.ps_data['T']


	
		### Update animation based on self.t
		t = self.ps_data['t']

		for i in range(len(self.ps_data['biomechanical'])):
			T = self.ps_data['biomechanical'][i].shape[0]
			self.ps_data['ps_biomechanical_list'][i].update_vertex_positions(self.ps_data['biomechanical'][i][self.ps_data['t']  %  T ])
			self.ps_data['ps_biomechanical_joints_list'][i].update_point_positions(self.ps_data['biomechanical_joints'][i][self.ps_data['t']  %  T ])

			if "Dynamics" in self.samples[i].mot_path:
				activation = Visualizer.load_mot(self.samples[i].mot_path)
				activation_muscle_name = [ x.replace('/activation', '') for x in activation['headers']]

				activations = time_normalization(activation['activations'],duration=196)
				activation_color = self.get_reds_color(activations)
				activation_color = dict([ (activation_muscle_name[j],activation_color[:,j,:3]) for j,muscle in enumerate(activation_muscle_name)])
			else:
				activation_color = dict([ (muscle,np.tile((np.array([255,0,0])/255).reshape(1,3),(T,1))) for muscle in self.ps_data['muscles'][i]])
       
			for j,muscle in enumerate(self.ps_data['muscles'][i]):
				# self.ps_data['ps_muscles_dict'][i][muscle].update_node_positions(self.ps_data['muscles'][i][muscle][self.ps_data['t']  %  T ])
				if muscle in self.ps_data['muscles'][i]:
					edges = np.array([ [i,i+1] for i in range(self.ps_data['muscles'][i][muscle].shape[1]-1)])
					ps_muscle = ps.register_curve_network(f"{muscle}-{i}",self.ps_data['muscles'][i][muscle][t],edges,color=activation_color[muscle][t])
					ps_muscle.add_to_group(self.ps_data[f'ps_muscles'][i])
		if not self.ps_data['is_paused']: 
			self.ps_data['t'] += 1 


		# Check keyboards for inputs
		
		# Check for spacebar press to toggle pause
		if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_Space)) or psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_Space)):
			
			self.ps_data['is_paused'] = not self.ps_data['is_paused']

		# Left arrow pressed
		if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_LeftArrow)) or psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_LeftArrow)):
			self.ps_data['t'] -= 1

		if psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_RightArrow)) or psim.IsKeyPressed(psim.GetKeyIndex(psim.ImGuiKey_RightArrow)):
			self.ps_data['t'] += 1

		# Input text
		changed, self.ps_data["ui_text"] = psim.InputText("- Coach Instructions", self.ps_data["ui_text"])

		# Create a floater to show the timestep and adject self.t accordingly
		changed, self.ps_data['t'] = psim.SliderInt("", self.ps_data['t'], v_min=0, v_max=self.ps_data['T'])
		psim.SameLine()

		# Create a render button which when pressed will create a .mp4 file
		if psim.Button("<"):
			self.ps_data['t'] -= 1
		
		psim.SameLine()
		if psim.Button("Play Video" if self.ps_data['is_paused'] else "Pause Video"):
			self.ps_data['is_paused'] = not self.ps_data['is_paused']

		psim.SameLine()
		if psim.Button(">"):
			self.ps_data['t'] += 1

		# psim.SameLine()
		if psim.Button("Render Video"):
			self.callback_render()        

		if(psim.TreeNode("Load other samples")):

			# psim.TextUnformatted("Load Optimized samples")

			changed = psim.BeginCombo("- Experiement", self.ps_data["session_options_selected"])
			if changed:				
				for val in self.ps_data["session_options"]:
					_, selected = psim.Selectable(val, selected=self.ps_data["session_options_selected"]==val)
					if selected:
						self.ps_data["session_options_selected"] = val

						sample_path = os.path.join(INPUT_DIR,f"OpenCapData_{self.ps_data['session_options_selected']}","MarkerData")
						other_session_trials_details = [ os.path.join(sample_path, x)  for x in os.listdir(sample_path)]
						other_session_trials_details = [OpenCapDataLoader.get_label(os.path.basename(x)) for x in other_session_trials_details if os.path.isfile(x)]

						self.ps_data['category_options'] = list(set([x[0] for x in other_session_trials_details]))
						self.ps_data['trial_options'] = [ x[1] for x in other_session_trials_details if x[0] == self.ps_data['category_options_selected']]

				psim.EndCombo()

			changed = psim.BeginCombo("- Trial", self.ps_data["trial_options_selected"])
			if changed:
				for val in self.ps_data["trial_options"]:
					_, selected = psim.Selectable(val, selected=self.ps_data["trial_options_selected"]==val)
					if selected:
						self.ps_data["trial_options_selected"] = val


						# Load info about other trial samples.
						sample_path = os.path.join(INPUT_DIR,f"OpenCapData_{self.ps_data['session_options_selected']}","MarkerData")
						other_session_trials_details = [ os.path.join(sample_path, x)  for x in os.listdir(sample_path)]
						other_session_trials_details = [OpenCapDataLoader.get_label(os.path.basename(x)) for x in other_session_trials_details if os.path.isfile(x)]

						self.ps_data['trial_options'] = [ x[1] for x in other_session_trials_details if x[0] == self.ps_data['category_options_selected']]
				psim.EndCombo()

			changed = psim.BeginCombo("- Suggested Samples", self.ps_data["retrieval_options_selected"])
			if changed:
				for val in self.ps_data["retrieval_options"]:
					_, selected = psim.Selectable(val, selected=self.ps_data["retrieval_options_selected"]==val)
					if selected:
						self.ps_data["retrieval_options_selected"] = val
				psim.EndCombo()


			
			if(psim.Button("Load Optimized samples")):
				sample_path = os.path.join(INPUT_DIR,f"OpenCapData_{self.ps_data['session_options_selected']}")
				sample_path = os.path.join(sample_path, "MarkerData")
				sample_path = os.path.join(sample_path,f"{self.ps_data['category_options_selected']}{self.ps_data['trial_options_selected']}.trc")
				# sample = load_subject(sample_path)

				retrieval_path = os.path.join(self.ps_data['retrieval_dir'], self.ps_data['retrieval_options_selected'])
				self.samples[-1] = load_retrived_samples(self.samples[-1],retrieval_path)
				self.update_smpl_multi_view_callback(self.samples)
			psim.TreePop()


		# psim.End()

	
	def update_smpl_multi_view_callback(self,samples,video_name=None):

		# For each sample
			# 1. Initialize ps_data to store all objects to render 
			# 2. Set camera  
			# 3. Shift on X-axis to render multiple views 
		

		for sample_ind, sample in enumerate(samples):
			assert hasattr(sample,'rgb'), "Error loading RGB Data. Don't know the camera details. Cannot render in multiple views"
			assert hasattr(sample,'osim'), "Error loading OSIM Data. Cannot render biomechanical data"

					
			# Initialize ps_data for polyscope
			if not hasattr(self,'ps_data')	:
				
				ps.init()
				ps.remove_all_structures()
				self.ps_data = {}
				self.ps_data['T'] = 196
				target = sample.joints_np
				self.ps_data['bbox'] = target.max(axis=(0,1)) - target.min(axis=(0,1))
				self.ps_data['object_position'] = sample.joints_np[0,0]


				self.ps_data['ps_biomechanical_list'] = []
				self.ps_data['ps_biomechanical_joints_list'] = []
				self.ps_data['ps_muscles'] = []
				self.ps_data['ps_muscles_dict'] = []
				
				self.ps_data['biomechanical'] = []
				self.ps_data['biomechanical_joints'] = []
				self.ps_data['muscles'] = []
				
				self.ps_data['com_curve'] = []

				# camera_shift_x = 0
				camera_shift_x = len(samples)//8
				camera_shift_x = self.ps_data['bbox'][0]*camera_shift_x
				print("Camera shift x",camera_shift_x)
				camera_position = np.array([camera_shift_x,-0.5*self.ps_data['bbox'][1],-5*self.ps_data['bbox'][0]]) + self.ps_data['object_position']
				look_at_position = np.array([camera_shift_x,-0.5*self.ps_data['bbox'][1],0]) + self.ps_data['object_position']
				print(camera_position,look_at_position)
				ps.look_at(camera_position,look_at_position)


			shift_x = sample_ind - len(samples)//2
			shift_x = -self.ps_data['bbox'][0]*shift_x*4
			print("Shift x:",shift_x)
			# shift_x = -1.5*shift_x





			if not hasattr(self,'ps_data') or 'ps_cams' not in self.ps_data:
				ps_cams = []
				# Set indivdual cameras 
				for i,camera in enumerate(sample.rgb.cameras): 
					intrinsics = ps.CameraIntrinsics(fov_vertical_deg=camera['fov_x'], fov_horizontal_deg=camera['fov_y'])
					# extrinsics = ps.CameraExtrinsics(mat=np.eye(4))
					extrinsics = ps.CameraExtrinsics(root=camera['position'], look_dir=camera['look_dir'], up_dir=camera['up_dir'])
					params = ps.CameraParameters(intrinsics, extrinsics)
					ps_cam = ps.register_camera_view(f"Cam{i}", params)
					# print("Camera:",params.get_view_mat())
					ps_cams.append(ps_cam)


				# Create random colors of each segment
				# colors = np.random.random((sample.segments.shape[0],3))
				# mesh_colors = np.zeros((verts.shape[0],3))
				# mesh_colors[:,1] = 0.3 # Default color is light blue
				# mesh_colors[:,2] = 1 # Default color is light blue
				# for i,segment in enumerate(sample.segments):
				# 	mesh_colors[segment[0]:segment[1]] = colors[i:i+1]

				# Map all polyscope objects to ps_data
		
				self.ps_data['ps_cams'] = ps_cams


			


			biomechanical = sample.osim.vertices + np.array([shift_x,0,0.0])*self.ps_data['bbox']
			biomechanical_joints = sample.osim.joints + np.array([shift_x,0,0.0])*self.ps_data['bbox']

			muscle_dict = {}
			for muscle_name in sample.osim.muscle_trajectory:        
				muscle_dict[muscle_name] = sample.osim.muscle_trajectory[muscle_name] + np.array([shift_x,0,0.0])*self.ps_data['bbox']




			rot_x = 0 
			rot_y = 90 if "t2m" in sample.osim_file or "mdm" in sample.osim_file else 0 
			rot_z = 0 


			R = sRotation.from_euler('xyz',[rot_x,rot_y,rot_z],degrees=True).as_matrix()

			biomechanical = (biomechanical  - biomechanical.mean((0,1),keepdims=True))@R.T + biomechanical.mean((0,1),keepdims=True) 
			biomechanical_joints = (biomechanical_joints  - biomechanical_joints.mean((0,1),keepdims=True) )@R.T + biomechanical_joints.mean((0,1),keepdims=True)
			for muscle_name in sample.osim.muscle_trajectory:        
				muscle_dict[muscle_name] = (muscle_dict[muscle_name] - muscle_dict[muscle_name].mean((0,1),keepdims=True))@R.T + muscle_dict[muscle_name].mean((0,1),keepdims=True)

			self.ps_data['biomechanical'].append(biomechanical)
			self.ps_data['biomechanical_joints'].append(biomechanical_joints)
			self.ps_data['muscles'].append(muscle_dict)




			name = os.path.basename(sample.osim_file).split('.')[0]
			# ps_biomechanical = ps.register_surface_mesh(f"{sample_ind}-{name} mesh",biomechanical[0],sample.osim.faces,transparency=0.5,color=np.array([127,127,255])/255,smooth_shade=True,material='wax')
			ps_biomechanical = ps.register_surface_mesh(f"{sample_ind}-{name} mesh",biomechanical[0],sample.osim.faces,transparency=0.5,color=np.array([255,255,255])/255,smooth_shade=True,material='wax')
			ps_biomechanical_joints = ps.register_point_cloud(f"{sample_ind}-{name} joints",biomechanical_joints[0],color=np.array([0,0,0]))

			edges = np.array([ [i,i+1] for i in range(biomechanical_joints.shape[0]-1)] + [[biomechanical_joints.shape[0]-1,0]]) 

			ps_com_curve = ps.register_curve_network(f"{sample_ind}-{name} com",biomechanical_joints[:,[1,6]].mean(axis=1),edges,color=np.array([255,255,0])/255)

			ps_muscles = ps.create_group(f"Muscles-{sample_ind}")
			ps_muscles_dict = {}
			for muscle in muscle_dict: 
				edges = np.array([ [i,i+1] for i in range(muscle_dict[muscle].shape[1]-1)])
				ps_muscle = ps.register_curve_network(f"{muscle}-{sample_ind}",muscle_dict[muscle][0],edges,color=np.array([255,0,0])/255)
				ps_muscle.add_to_group(ps_muscles)			
				ps_muscles_dict[muscle] = ps_muscle 
			ps_muscles.set_enabled(False)
	

			self.ps_data['ps_biomechanical_list'].append(ps_biomechanical)
			self.ps_data['ps_biomechanical_joints_list'].append(ps_biomechanical_joints)	
			self.ps_data['ps_muscles'].append(ps_muscles)
			self.ps_data['ps_muscles_dict'].append(ps_muscles_dict)

			# Find the timestep with knee flexion and set it as the initial timestep
			deepest_squat_index = biomechanical_joints[:,0,1].argmin()

			self.ps_data['ps_biomechanical_list'][-1].update_vertex_positions(biomechanical[deepest_squat_index])
			self.ps_data['ps_biomechanical_joints_list'][-1].update_point_positions(biomechanical_joints[deepest_squat_index])


			# Map all the rendering information
			self.ps_data['video_name'] = video_name
			self.ps_data['label'] = sample.label
			self.ps_data['recordAttempt'] = sample.recordAttempt
			self.ps_data['fps'] = sample.fps

			# Animation details  
			self.ps_data['t'] = 0 
			self.ps_data['T'] = max(self.ps_data['T'], sample.osim.vertices.shape[0])
			self.ps_data['is_paused'] = False
			self.ps_data['ui_text'] = "Enter Instructions here"


			self.ps_data['session_options_selected'] = sample.openCapID
			self.ps_data['session_options'] = [ x.replace("OpenCapData_","") for x in  os.listdir(INPUT_DIR)]

			# Load info about other trial samples.
			other_session_trials_details = [ os.path.join(os.path.dirname(sample.osim_file), x)  for x in os.listdir(os.path.dirname(sample.osim_file))]
			other_session_trials_details = [OpenCapDataLoader.get_label(os.path.basename(x)) for x in other_session_trials_details if os.path.isfile(x)]
			
			self.ps_data['category_options_selected'] = sample.label 
			self.ps_data['category_options'] = list(set([x[0] for x in other_session_trials_details]))
			

			self.ps_data['trial_options_selected'] = sample.recordAttempt_str
			self.ps_data['trial_options'] = [ x[1] for x in other_session_trials_details if x[0] == self.ps_data['category_options_selected']]

			self.ps_data['retrieval_options_selected'] = os.path.basename(sample.osim_file)
			self.ps_data['retrieval_dir'] = os.path.dirname(sample.osim_file)
			self.ps_data['retrieval_options'] = os.listdir(self.ps_data['retrieval_dir'])  
		


		# Set camera

		# camera_position = np.array([0,0,3*self.ps_data['bbox'][0]])
		# camera_position = np.array([7*self.ps_data['bbox'][0],0.0*self.ps_data['bbox'][1],0]) + self.ps_data['object_position']



		# Take a screenshot of the initial setup 
		# ps.show()
		if video_name is not None:
			image_path = video_name.replace('.mp4', "_initial.png")
			# print(f"Saving plot to :{image_path}")	
			ps.set_screenshot_extension(".png")
			ps.screenshot(image_path,transparent_bg=True) 





	def callback_render(self):



		video_name = self.ps_data['video_name']

		if video_name is None: 
			ps.warning("Location to render not specefied. Setting to <current working directory>/render as default")
			video_name = os.path.join(os.getcwd(),"render")

		if not video_name.endswith('.mp4'):
			video_name = video_name + "mp4"





		image_dir = os.path.join(video_name.replace('.mp4', "_images_transparent"))
		os.makedirs(image_dir,exist_ok=True)
		print(f'Rendering images:')
		for i in tqdm(range(self.ps_data['T'])):
			
			for j in range(len(self.ps_data['ps_biomechanical_list'])):
				self.ps_data['ps_biomechanical_list'][j].update_vertex_positions(self.ps_data['biomechanical'][j][i])
				self.ps_data['ps_biomechanical_joints_list'][j].update_point_positions(self.ps_data['biomechanical_joints'][j][i])



			image_path = os.path.join(image_dir,f"{os.path.basename(video_name).replace('.mp4','')}_{i}.png")
			# print(f"Saving plot to :{image_path}")	
			ps.set_screenshot_extension(".png")
			ps.screenshot(image_path,transparent_bg=True)





		image_dir = os.path.join(video_name.replace('.mp4', "_images"))
		os.makedirs(image_dir,exist_ok=True)
		print(f'Rendering images:')
		for i in tqdm(range(self.ps_data['T'])):
			
			for j in range(len(self.ps_data['ps_biomechanical_list'])):
				self.ps_data['ps_biomechanical_list'][j].update_vertex_positions(self.ps_data['biomechanical'][j][i])
				self.ps_data['ps_biomechanical_joints_list'][j].update_point_positions(self.ps_data['biomechanical_joints'][j][i])



			image_path = os.path.join(image_dir,f"smpl_{i}.png")
			# print(f"Saving plot to :{image_path}")	
			ps.set_screenshot_extension(".png")
			ps.screenshot(image_path,transparent_bg=False)
			

		image_path = os.path.join(image_dir,f"smpl_\%d.png")
		video_name = os.path.abspath(video_name)
		palette_path = os.path.join(image_dir,f"smpl.png")
		frame_rate = self.ps_data['fps']//2 # Slowed down by 2x
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -vf palettegen {palette_path}")
		os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse 	-q:v 5 {video_name}")	
		# os.system(f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_name.replace('mp4','gif')}")	

		print(f"Running Command:",f"ffmpeg -y -framerate {frame_rate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_name}")





	# Initialize 3D objects from a sample and set callback for
	def render_smpl_multi_view_callback(self,samples,video_name=None):
		self.samples = samples
		self.update_smpl_multi_view_callback(samples,video_name=video_name)
		# ps.show()
		ps.set_user_callback(self.callback)
		ps.show()	



def render(samples, vis,video_name=None): 
	"""
		Render dataset samples 
			

		@params
			sample_path: Filepath of input
			video_dir: Folder to store Render results for the complete worflow  
		
			
		Load input (currently .trc files) and save all the rendering videos + images (retargetting to smp, getting input text, per frame annotations etc.) 
	"""

	
	if video_name is not None:
		vis.update_smpl_multi_view_callback(samples,video_name=video_name)
		vis.callback_render()	
		assert os.path.isfile(video_name), f"FFMPEG unable to render video:{video_name}"
	else:
		# Visualize each view  
		vis.render_smpl_multi_view_callback(samples,video_name=None)



def load_subject(sample_path,retrieval_path=None):
	sample = OpenCapDataLoader(sample_path)
	
	# Load Video
	sample.rgb = MultiviewRGB(sample)

	print(f"Session ID: {sample.name} SubjectID:{sample.rgb.session_data['subjectID']} Action:{sample.label}")

	osim_path = os.path.dirname(os.path.dirname(sample.sample_path)) 
	osim_path = os.path.join(osim_path,'OpenSimData','Model', 'LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim')
	osim_geometry_path = os.path.join(DATA_DIR,'OpenCap_LaiArnoldModified2017_Geometry')


	###################### Subject Details ####################################################	
	mot_path = os.path.dirname(os.path.dirname(sample.sample_path))
	mot_path = os.path.join(mot_path,'OpenSimData','Kinematics',sample.label+ sample.recordAttempt_str + '.mot')
	mot_path = os.path.abspath(mot_path)
	print("Loading User motion file:",mot_path)
	sample.osim_file = mot_path


	samples = []
	# Load Segments
	if os.path.exists(os.path.join(DATA_DIR,"squat-segmentation-data", sample.openCapID+'.npy')):
		segments = np.load(os.path.join(DATA_DIR,"squat-segmentation-data", sample.openCapID+'.npy'),allow_pickle=True).item()
		if os.path.basename(sample.sample_path).split('.')[0] in segments:
			segments = segments[sample.label+ sample.recordAttempt_str]


			for segment in segments:	
				cur_sample = copy.deepcopy(sample)
				cur_sample.joints_np = cur_sample.joints_np[segment[0]:segment[1]]
				cur_sample.osim = OSIMSequence.from_files(osim_path, mot_path, geometry_path=osim_geometry_path,ignore_fps=True, start_frame=segment[0], end_frame=segment[1])

				samples.append(cur_sample)
				break

	if len(samples) == 0:
		sample.osim = OSIMSequence.from_files(osim_path, mot_path, geometry_path=osim_geometry_path,ignore_fps=True )
		samples.append(sample)

	return samples

def load_retrived_samples(session, retrieval_path): 
	###################### GENERATION DETAILS ####################################################

	# mot_path = os.path.dirname(os.path.dirname(sample.sample_path))
	# mot_path = os.path.join(mot_path,'OpenSimData','VQVAE7_Temporal_Kinematics',sample.label+ sample.recordAttempt_str + '.mot')
	# print("Loading Reconstrction file:",mot_path)

	# sample.osim_pred = OSIMSequence.from_files(osim_path, mot_path, geometry_path=osim_geometry_path,ignore_fps=True )	

	# mot_path = "/media/shubh/Elements/RoseYu/UCSD-OpenCap-Fitness-Dataset/MCS_DATA/mot_visualization/constrained_mot_0.002/12.mot"
	# mot_path = "/media/shubh/Elements/RoseYu/UCSD-OpenCap-Fitness-Dataset/MCS_DATA/mot_visualization/normal_latents_196/entry_2.mot"

	# mot_path = "/media/shubh/Elements/RoseYu/UCSD-OpenCap-Fitness-Dataset/MCS_DATA/mot_visualization/normal_latents_temporal_consistency_v2/entry_9.mot"


	assert retrieval_path and os.path.isfile(retrieval_path), f"Unable to load .mot file:{retrieval_path}" 

	mot_path = os.path.abspath(retrieval_path)
	print("Loading Generatrion file:",mot_path)
	

	trc_path = os.path.join(DATA_DIR,"Data", session, "MarkerData")
	trc_file = [os.path.join(trc_path,x) for x in os.listdir(trc_path) if  'sqt' in x.lower()  and x.endswith('.trc')  ][0]
	sample = OpenCapDataLoader(trc_file)
	
	sample.mot_path = mot_path	
 
	# Load Video
	sample.rgb = MultiviewRGB(sample)

	


	osim_path = os.path.join(DATA_DIR,"Data", session, "OpenSimData","Model","LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim") 
	osim_geometry_path = os.path.join(DATA_DIR,'OpenCap_LaiArnoldModified2017_Geometry')

	sample.osim = OSIMSequence.from_files(osim_path, mot_path, geometry_path=osim_geometry_path,ignore_fps=True )	
	print("MOT DATA:",sample.osim.motion.shape)
	print("Pelivs:",np.rad2deg(sample.osim.motion[::10,1:3]))
	print("KNEE Left:",np.rad2deg(sample.osim.motion[::10,10]))
	print("TIME:",sample.osim.motion[::10,0])
	# sample.osim.vertices[:,:,2] -= 1	
	sample.osim_file = retrieval_path
	return sample


def load_dynamics_data(retrieval_path): 
	###################### GENERATION DETAILS ####################################################

	# mot_path = os.path.dirname(os.path.dirname(sample.sample_path))
	# mot_path = os.path.join(mot_path,'OpenSimData','VQVAE7_Temporal_Kinematics',sample.label+ sample.recordAttempt_str + '.mot')
	# print("Loading Reconstrction file:",mot_path)

	# sample.osim_pred = OSIMSequence.from_files(osim_path, mot_path, geometry_path=osim_geometry_path,ignore_fps=True )	

	# mot_path = "/media/shubh/Elements/RoseYu/UCSD-OpenCap-Fitness-Dataset/MCS_DATA/mot_visualization/constrained_mot_0.002/12.mot"
	# mot_path = "/media/shubh/Elements/RoseYu/UCSD-OpenCap-Fitness-Dataset/MCS_DATA/mot_visualization/normal_latents_196/entry_2.mot"

	# mot_path = "/media/shubh/Elements/RoseYu/UCSD-OpenCap-Fitness-Dataset/MCS_DATA/mot_visualization/normal_latents_temporal_consistency_v2/entry_9.mot"


	assert retrieval_path and os.path.isfile(retrieval_path), f"Unable to load .mot file:{retrieval_path}" 


	mot_path = os.path.abspath(retrieval_path)
	print("Loading Generatrion file:",mot_path)


	trc_path = os.path.join(DATA_DIR,"Data", session, "MarkerData")
	trc_file = [os.path.join(trc_path,x) for x in os.listdir(trc_path) if  'sqt' in x.lower()  and x.endswith('.trc')  ][0]
	sample = OpenCapDataLoader(trc_file)
	
	# Load Video
	sample.rgb = MultiviewRGB(sample)


	osim_path = os.path.join(DATA_DIR,"Data", session, "OpenSimData","Model","LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim") 
	osim_geometry_path = os.path.join(DATA_DIR,'OpenCap_LaiArnoldModified2017_Geometry')

	sample.osim = OSIMSequence.from_files(osim_path, mot_path, geometry_path=osim_geometry_path,ignore_fps=True )	
	print("MOT DATA:",sample.osim.motion.shape)
	print("Pelivs:",np.rad2deg(sample.osim.motion[::10,1:3]))
	print("KNEE Left:",np.rad2deg(sample.osim.motion[::10,10]))
	print("TIME:",sample.osim.motion[::10,0])
	# sample.osim.vertices[:,:,2] -= 1	
	sample.osim_file = retrieval_path
	return sample

if __name__ == "__main__": 

	import copy


	# compare_files = ["/media/shubh/Elements/RoseYu/UCSD-OpenCap-Fitness-Dataset/MCS_DATA/Data/d66330dc-7884-4915-9dbb-0520932294c4/MarkerData/SQT01.trc",
	# 			      "MCS_DATA/LIMO/ComAcc/mot_visualization/latents_subject_run_000cffd9-e154-4ce5-a075-1b4e1fd66201/entry_17_ComAcc.mot", 
	# 				   "MCS_DATA/LIMO/FinalFinalHigh/mot_visualization/latents_subject_run_d2020b0e-6d41-4759-87f0-5c158f6ab86a/entry_19_FinalFinalHigh.mot"]

	compare_files = sys.argv[1:]
	


	if len(sys.argv) == 0: # Skipping for now. Will fix later to generate videos   
		render_dataset()
	else:
		samples = [] 
		for i in range(0, min(len(compare_files),4)  ): 
			# file_path = sys.argv[i]
			file_path = compare_files[i]
			if file_path.endswith('.trc'):
				sample = load_subject(file_path)
				samples.extend(sample)
			elif 'OpenSimData/Dynamics' in file_path: # For Dynamics data
							
				session = file_path
				for i in range(4):
					session = os.path.dirname(session)
				session = os.path.basename(session)
				print(session)
				sample = load_retrived_samples(session, file_path)
				samples.append(sample)


			elif file_path.endswith('.mot'): # For baseline + generated results 
				session = os.path.basename(os.path.dirname(file_path))
				session = session.replace("latents_subject_run_","")
				sample = load_retrived_samples(session,file_path)

				samples.append(sample)
		vis = Visualizer()
		video_name = None
		video_name = sys.argv[-1] if sys.argv[-1].endswith('mp4') else None
		render(samples,vis,video_name)

