import sys 
import os 
sys.path.insert(0,os.getcwd())


from models.rotation2xyz import Rotation2xyz
import numpy as np
from trimesh import Trimesh
import os
os.environ['PYOPENGL_PLATFORM'] = "osmesa"

import torch
from visualize.simplify_loc2rot import joints2smpl
import pyrender
import matplotlib.pyplot as plt

import io
import imageio
from shapely import geometry
import trimesh
from pyrender.constants import RenderFlags
import math
# import ffmpeg
from PIL import Image

from utils.motion_process import recover_from_ric

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P

def render(motions, outdir='test_vis', device_id=0, name=None, pred=True):
    frames, njoints, nfeats = motions.shape
    MINS = motions.min(axis=0)[0].min(axis=0)[0]
    MAXS = motions.max(axis=0)[0].max(axis=0)[0]

    # print(f'MIN: {MINS}, MAX: {MAXS}')

    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset
    trajec = motions[:, 0, [0, 2]]

    j2s = joints2smpl(num_frames=frames, device_id=0, cuda=True)
    rot2xyz = Rotation2xyz(device=torch.device("cuda:0"))
    faces = rot2xyz.smpl_model.faces

    if (not os.path.exists(outdir + name+'_pred.pt') and pred) or (not os.path.exists(outdir + name+'_gt.pt') and not pred): 
        print(f'Running SMPLify, it may take a few minutes.')
        motion_tensor, opt_dict = j2s.joint2smpl(motions)  # [nframes, njoints, 3]

        vertices = rot2xyz(torch.tensor(motion_tensor).clone(), mask=None,
                                        pose_rep='rot6d', translation=True, glob=True,
                                        jointstype='vertices',
                                        vertstrans=True)

        if pred:
            torch.save(vertices, outdir + name+'_pred.pt')
        else:
            torch.save(vertices, outdir + name+'_gt.pt')
    else:
        if pred:
            vertices = torch.load(outdir + name+'_pred.pt')
        else:
            vertices = torch.load(outdir + name+'_gt.pt')
    frames = vertices.shape[3] # shape: 1, nb_frames, 3, nb_joints
    print (vertices.shape)
    MINS = torch.min(torch.min(vertices[0], axis=0)[0], axis=1)[0]
    MAXS = torch.max(torch.max(vertices[0], axis=0)[0], axis=1)[0]
    # vertices[:,:,1,:] -= MINS[1] + 1e-5


    out_list = []
    
    minx = MINS[0] - 0.5
    maxx = MAXS[0] + 0.5
    minz = MINS[2] - 0.5 
    maxz = MAXS[2] + 0.5
    polygon = geometry.Polygon([[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]])
    polygon_mesh = trimesh.creation.extrude_polygon(polygon, 1e-5)

    vid = []
    for i in range(frames):
        if i % 10 == 0:
            print(i)

        mesh = Trimesh(vertices=vertices[0, :, :, i].squeeze().tolist(), faces=faces)

        base_color = (0.11, 0.53, 0.8, 0.5)
        ## OPAQUE rendering without alpha
        ## BLEND rendering consider alpha 
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=base_color
        )


        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        polygon_mesh.visual.face_colors = [0, 0, 0, 0.21]
        polygon_render = pyrender.Mesh.from_trimesh(polygon_mesh, smooth=False)

        bg_color = [1, 1, 1, 0.8]
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))
        
        sx, sy, tx, ty = [0.75, 0.75, 0, 0.10]

        camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))

        light = pyrender.DirectionalLight(color=[1,1,1], intensity=300)

        scene.add(mesh)

        c = np.pi / 2

        scene.add(polygon_render, pose=np.array([[ 1, 0, 0, 0],

        [ 0, np.cos(c), -np.sin(c), MINS[1].cpu().numpy()],

        [ 0, np.sin(c), np.cos(c), 0],

        [ 0, 0, 0, 1]]))

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [0, 1, 1]
        scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = [1, 1, 2]
        scene.add(light, pose=light_pose.copy())


        c = -np.pi / 6

        scene.add(camera, pose=[[ 1, 0, 0, (minx+maxx).cpu().numpy()/2],

                                [ 0, np.cos(c), -np.sin(c), 1.5],

                                [ 0, np.sin(c), np.cos(c), max(4, minz.cpu().numpy()+(1.5-MINS[1].cpu().numpy())*2, (maxx-minx).cpu().numpy())],

                                [ 0, 0, 0, 1]
                                ])
        
        # render scene
        r = pyrender.OffscreenRenderer(960, 960)

        color, _ = r.render(scene, flags=RenderFlags.RGBA)
        # Image.fromarray(color).save(outdir+name+'_'+str(i)+'.png')

        vid.append(color)

        r.delete()

    out = np.stack(vid, axis=0)
    if pred:
        imageio.mimsave(outdir + name+'_pred.gif', out, fps=20)
    else:
        imageio.mimsave(outdir + name+'_gt.gif', out, fps=20)


import polyscope as ps
import polyscope.imgui as psim

class PolyScopeVisualizer:
    def __init__(self,render_path=None,exp_dir='./output'):
        

        ps.init()
        ps.set_verbosity(2)        
        ps.set_enable_render_error_checks(True)


        self.t = 0
        self.T = np.inf
    
        # Experiments_dirs
        self.exp_dir = exp_dir
        self.exps = [file for file in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, file))  and 'LIMO' in file]

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
            "category_options_selected": self.categories[1],

            "rank": 1,

            "is_paused": False
        }



        self.render_path = render_path




    def callback(self):
        
        ########### Checks ############
        # Ensure self.t lies between 
        self.t %= self.T






        ### Update animation based on self.t
        if hasattr(self, 'smpl_skeleton'):
            self.smpl_skeleton.update_node_positions(self.motions[self.t])
        
        
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
        changed, self.t = psim.SliderInt("", self.t, v_min=0, v_max=self.T)
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
            self.render()        

        if(psim.TreeNode("Load Experiment")):

            # psim.TextUnformatted("Load Optimized samples")

            changed = psim.BeginCombo("- Experiement", self.polyscope_scene["experiment_options_selected"])
            if changed:
                for val in self.polyscope_scene["experiment_options"]:
                    _, selected = psim.Selectable(val, selected=self.polyscope_scene["experiment_options_selected"]==val)
                    if selected:
                        self.polyscope_scene["experiment_options_selected"] = val
                psim.EndCombo()

            changed = psim.BeginCombo("- Category", self.polyscope_scene["category_options_selected"])
            if changed:
                for val in self.polyscope_scene["category_options"]:
                    _, selected = psim.Selectable(val, selected=self.polyscope_scene["category_options_selected"]==val)
                    if selected:
                        self.polyscope_scene["category_options_selected"] = val
                psim.EndCombo()



            changed, new_rank = psim.InputInt("- rank", self.polyscope_scene["rank"], step=1, step_fast=10) 
            if changed: 
                self.polyscope_scene["rank"] = new_rank # Only change values when button is pressed. Otherwise will be continously update like self.t 
                
                if self.polyscope_scene["rank"] > 100:
                    self.polyscope_scene['rank'] = 100
                elif self.polyscope_scene["rank"] < 1: 
                    self.polyscope_scene['rank'] = 1 
                else: 
                    pass

            
            if(psim.Button("Load Optimized samples")):
                self.update_skeleton()
            psim.TreePop()


        psim.End()

        # == Set parameters

        # These commands allow the user to adjust the value of variables.
        # It is important that we assign the return result to the variable to
        # update it. 
        # For most elements, the return is actually a tuple `(changed, newval)`, 
        # where `changed` indicates whether the setting was modified on this 
        # frame, and `newval` gives the new value of the variable (or the same 
        # old value if unchanged).
        #
        # For numeric inputs, ctrl-click on the box to type in a value.


        # Checkbox
        # changed, self.polyscope_scene["is_true1"] = psim.Checkbox("flag1", self.polyscope_scene["is_true1"]) 
        # if(changed): # optionally, use this conditional to take action on the new value
            # psim.SetWindowSize((0.0,200.0),1)
            # psim.SetWindowPos("Some view",(0.0,0.0),1)

            # print("Checkbox changed to ", self.polyscope_scene["is_true1"])
            # pass 
        # psim.SameLine() 
        # changed, self.polyscope_scene["is_true2"] = psim.Checkbox("flag2", self.polyscope_scene["is_true2"]) 

        # Input ints



        # Input floats using two different styles of widget
        # changed, self.polyscope_scene["ui_float1"] = psim.InputFloat("ui_float1", self.polyscope_scene["ui_float1"]) 
        # psim.SameLine() 


        # Input colors
        # changed, self.polyscope_scene["ui_color3"] = psim.ColorEdit3("ui_color3", self.polyscope_scene["ui_color3"])
        # psim.SameLine() 
        # changed, self.polyscope_scene["ui_color4"] = psim.ColorEdit4("ui_color4", self.polyscope_scene["ui_color4"])


        # Combo box to choose from options
        # There, the options are a list of strings in `ui_options`,
        # and the currently selected element is stored in `ui_options_selected`.
        # psim.PushItemWidth(200)

        # psim.PopItemWidth()


        # Use tree headers to logically group options

        # This a stateful option to set the tree node below to be open initially.
        # The second argument is a flag, which works like a bitmask.
        # Many ImGui elements accept flags to modify their behavior.
        # psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)

        # # The body is executed only when the sub-menu is open. Note the push/pop pair!


        # psim.PopItemWidth()


    def update_skeleton(self):
        filepath = os.path.join(self.exp_dir,self.polyscope_scene['experiment_options_selected'])
        filepath = os.path.join(filepath,'category_' + self.polyscope_scene['category_options_selected'].replace('fast', 'full').replace(' ', '_'))
        filepath = os.path.join(filepath, f"entry_{self.polyscope_scene['rank']-1}.npy") 
        if not os.path.isfile(filepath):
            ps.error(f"Unable to locate:{filepath}");
            return


        motions = np.load(filepath)        
        num_joints = 22
        motions = recover_from_ric(torch.from_numpy(motions).float().cuda(), num_joints)
        motions = motions.detach().cpu().numpy()
        motions[:,:,2] *= -1 # Replace z-axis with -z-axis.

        self.motions = motions
        self.T = motions.shape[0]

        print("Successfully loaded", filepath)

    def render_skeleton(self,savepath):
        self.update_skeleton()

        bone_array = [0,0, 0, 0,1, 2, 3, 4, 5, 6, 7,8,9,9,9,12,13,14,16,17,18,19,20,21]
        smpl_bone_array = np.array([[i,p] for i,p in enumerate(bone_array)])

        ps.init()
        self.smpl_skeleton = ps.register_curve_network("My skelton", self.motions[0], smpl_bone_array[:22])

        self.render_path = savepath
        ps.set_user_callback(self.callback)

        ps.show()

    

    def render(self): 
        
        os.makedirs('/tmp/skeleton/',exist_ok=True)
        for i in range(self.motions.shape[0]):
            self.smpl_skeleton.update_node_positions(self.motions[i])
            ps.screenshot(f"/tmp/skeleton/{i}.png")

        os.system(f"ffmpeg -y -i /tmp/skeleton/%d.png -pix_fmt yuv420p {self.render_path}.mp4")



if __name__ == "__main__":
    import sys
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("--filedir", type=str, default=None, help='motion npy file dir')
    parser.add_argument("--out-dir", type=str, default="output-viz", help='Directory containing all experiements')
    parser.add_argument('--motion-list', default=None, nargs="+", type=str, help="motion name list")


    # # Create a private key for connecting to the server using ssh-agent or PuYYYgen https://stackoverflow.com/questions/2224066/how-to-convert-ssh-keypairs-generated-using-puttygen-windows-into-key-pairs-us
    # parser.add_argument('--server-key', default=None, type=str, help="Private key for the server")
    args = parser.parse_args()

    
    # sys.path.append(os.path.dirname(os.path.abspath(__file__)))


    # from server import NorthUCSDServer
    
    # ssh_client = NorthUCSDServer(key_filename=args.server_key)

    # stdin, stdout, stderr = ssh_client.exec_command('ls')
    # print(stdout.readlines())
    # ssh_client.close()

    # def sync(): 
    #     ssh_client.sync_from_remote()
        
        
        
        
        
    #     ssh_client.sync_to_remote()



    filename_list = args.motion_list
    filedir = args.filedir
    
    vis = PolyScopeVisualizer(exp_dir=args.out_dir)

    for filename in filename_list:


        vis.polyscope_scene["rank"] = int(filename.split('_')[-1])
        vis.polyscope_scene["category_options_selected"] = os.path.basename(filedir).replace("category_","")
        vis.polyscope_scene["experiment_options_selected"] = os.path.basename(os.path.dirname(filedir))

        vis.render_skeleton(os.path.join(filedir,filename))
        # render(motions, outdir=filedir, device_id=0, name=filename, pred=True)

        # motions = np.load(filedir + filename+'_gt.npy')
        # print('gt', motions.shape, filename)
        # render(motions[0], outdir=filedir, device_id=0, name=filename, pred=False)


    