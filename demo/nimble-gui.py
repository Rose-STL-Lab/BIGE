import os
import sys
import time
import numpy as np 
import torch
from easydict import EasyDict

import nimblephysics as nimble
from nimblephysics import NimbleGUI
from typing import Dict, Tuple, List
from website_sample_loader import WebsiteSample
from file_locator import FileLocator

# Sample loading imports - can be moved to separate file later
dir_path = os.path.dirname(os.path.abspath(__file__))
UCSD_OpenCap_Fitness_Dataset_path = os.path.abspath(os.path.join(dir_path,'..', '..', 'UCSD-OpenCap-Fitness-Dataset' , 'src'))
sys.path.append(UCSD_OpenCap_Fitness_Dataset_path)
from utils import DATA_DIR


class BigeDatasetVisualizer():
    def __init__(self):
        super().__init__()
        
        self.sample = None

        self.samples = []
        self.gui = None
        self.world = None
        
        self.file_locator = FileLocator()

    def ensure_geometry(self, geometry: str):
        if geometry is None:
            # Check if the "./Geometry" folder exists, and if not, download it
            if not os.path.exists('./Geometry'):
                print('Downloading the Geometry folder from https://addbiomechanics.org/resources/Geometry.zip')
                exit_code = os.system('wget https://addbiomechanics.org/resources/Geometry.zip')
                if exit_code != 0:
                    print('ERROR: Failed to download Geometry.zip. You may need to install wget. If you are on a Mac, '
                          'try running "brew install wget"')
                    return False
                os.system('unzip ./Geometry.zip')
                os.system('rm ./Geometry.zip')
            geometry = './Geometry'
        print('Using Geometry folder: ' + geometry)
        geometry = os.path.abspath(geometry)
        if not geometry.endswith('/'):
            geometry += '/'
        return geometry

    def create_default_args(self):
        """Create default arguments using EasyDict instead of argparse"""
        args = EasyDict()
        args.geometry_folder = os.path.join(DATA_DIR, "OpenCap_LaiArnoldModified2017_Geometry")
        # Set all the default values from the original parser arguments
        args.dataset_home = '../data'
        args.model_type = 'feedforward'
        args.output_data_format = 'all_frames'  # choices: ['all_frames', 'last_frame']
        args.device = 'cpu'
        args.checkpoint_dir = '../checkpoints'
        args.history_len = 50
        args.stride = 5
        args.dropout = False  # store_true becomes False by default
        args.dropout_prob = 0.5
        args.hidden_dims = [512, 512]
        args.batchnorm = False  # store_true becomes False by default
        args.activation = 'sigmoid'
        args.batch_size = 32
        args.short = False  # store_true becomes False by default
        args.predict_grf_components = [i for i in range(6)]
        args.predict_cop_components = [i for i in range(6)]
        args.predict_moment_components = [i for i in range(6)]
        args.predict_wrench_components = [i for i in range(12)]    
        return args
    
    def create_gui(self):
        world = nimble.simulation.World()
        world.setGravity([0, -9.81, 0])

        gui = NimbleGUI(self.world)
        gui.serve(8000)

        return world, gui

    def run(self, args):
        """
        Iterate over all *.b3d files in a directory hierarchy,
        compute file hash, and move to train or dev directories.
        """

        geometry = self.ensure_geometry(args.geometry_folder)
        self.world, self.gui = self.create_gui()


        frame: int = 0
        playing: bool = True
        num_frames = 196
        if num_frames == 0:
            print('No frames in dataset!')
            exit(1)

        # Add keypress  listener to control playback and space 
        def onKeyPress(key):
            nonlocal playing
            nonlocal frame
            if key == ' ':
                playing = not playing
            elif key == 'e':
                frame += 1
                if frame >= num_frames - 5:
                    frame = 0
            elif key == 'a':
                frame -= 1
                if frame < 0:
                    frame = num_frames - 5

        self.gui.nativeAPI().registerKeydownListener(onKeyPress)

        # Test relevant files
        compare_files = [
                    "LIMO/FinalFinalHigh/mot_visualization/latents_subject_run_d2020b0e-6d41-4759-87f0-5c158f6ab86a/entry_19_FinalFinalHigh.mot",
                    "Data/d66330dc-7884-4915-9dbb-0520932294c4/OpenSimData/Kinematics/SQT01.mot",
                    # "Data/d66330dc-7884-4915-9dbb-0520932294c4/MarkerData/SQT01.trc",
                    # "Data/002392d8-b28a-46e9-84ee-65053ec83739/OpenSimData/Dynamics/Sqt02_segment_1/kinematics_activations_Sqt02_segment_1_muscle_driven.mot",
                    # "t2m_baseline/015b7571-9f0b-4db4-a854-68e57640640d/0d9e84e9-57a4-4534-aee2-0d0e8d1e7c45_SQT01_0_pred_3_radians.mot",
                    # "LIMO/ComAcc/mot_visualization/latents_subject_run_000cffd9-e154-4ce5-a075-1b4e1fd66201/entry_17_ComAcc.mot", 
                    ]

        compare_files = [os.path.join(DATA_DIR, compare_file) for compare_file in compare_files]
            
        # Use the WebsiteSample class to load samples
        web_samples = [WebsiteSample(f_ind,file_path, self.gui, self.file_locator) for f_ind, file_path in enumerate(compare_files)]
        # Add all skeletons to the shared world
        for web_sample in web_samples:
            if hasattr(web_sample, 'skeleton'):
                self.world.addSkeleton(web_sample.skeleton)

        def onTick(now):
            with torch.no_grad():
                nonlocal frame
                for sample_ind, web_sample in enumerate(web_samples):
                    sample_frame = frame % web_sample.sample.osim.motion.shape[0]
                    motion = web_sample.sample.osim.motion[sample_frame, :].copy()
                    # motion[3] += sample_ind - len(web_samples) / 2  # Offset each sample for visibility
                    web_sample.skeleton.setPositions(motion)

                # Render the entire world
                self.gui.nativeAPI().renderWorld(self.world, prefix="world")


                if playing:
                    frame += 1
                    # if frame >= num_frames - 5:
                    #     frame = 0


        ticker: nimble.realtime.Ticker = nimble.realtime.Ticker(0.04)
        ticker.registerTickListener(onTick)
        ticker.start()
        # Don't immediately exit while we're serving
        self.gui.blockWhileServing()
        return True


if __name__ == '__main__':
    website = BigeDatasetVisualizer()
    args = website.create_default_args()
    website.run(args)