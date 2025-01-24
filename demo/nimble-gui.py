import os
import sys
import copy
import time
import torch
import logging
import argparse
import numpy as np

import nimblephysics as nimble
from nimblephysics import NimbleGUI

from typing import Dict, Tuple, List

file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
UCSD_OpenCap_Fitness_Dataset_path = os.path.join(dir_path,'..', '..', 'UCSD-OpenCap-Fitness-Dataset' , 'src')
UCSD_OpenCap_Fitness_Dataset_path = os.path.abspath(UCSD_OpenCap_Fitness_Dataset_path)
sys.path.append(UCSD_OpenCap_Fitness_Dataset_path)
print(UCSD_OpenCap_Fitness_Dataset_path)



from utils import DATA_DIR 
from dataloader import OpenCapDataLoader,MultiviewRGB
from smpl_loader import SMPLRetarget

# from osim import OSIMSequence
# Load LaiArnoldModified2017
from osim import OSIMSequence

class VisualizeCommand():
    def __init__(self):
        super().__init__()

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

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('visualize', help='Visualize the performance of a model on dataset.')

        subparser.add_argument('--dataset-home', type=str, default='../data',
                               help='The path to the AddBiomechanics dataset.')
        subparser.add_argument('--model-type', type=str, default='feedforward', help='The model to train.')
        subparser.add_argument('--output-data-format', type=str, default='all_frames', choices=['all_frames', 'last_frame'], 
                               help='Output for all frames in a window or only the last frame.')
        subparser.add_argument('--device', type=str, default='cpu', help='Where to run the code, either cpu or gpu.')
        subparser.add_argument('--checkpoint-dir', type=str, default='../checkpoints',
                               help='The path to a model checkpoint to save during training. Also, starts from the '
                                    'latest checkpoint in this directory.')
        subparser.add_argument('--geometry-folder', type=str, default=None,
                               help='Path to the Geometry folder with bone mesh data.')
        subparser.add_argument('--history-len', type=int, default=50,
                               help='The number of timesteps of context to show when constructing the inputs.')
        subparser.add_argument('--stride', type=int, default=5,
                               help='The number of timesteps of context to show when constructing the inputs.')
        subparser.add_argument('--dropout', action='store_true', help='Apply dropout?')
        subparser.add_argument('--dropout-prob', type=float, default=0.5, help='Dropout prob')
        subparser.add_argument('--hidden-dims', type=int, nargs='+', default=[512, 512],
                               help='Hidden dims across different layers.')
        subparser.add_argument('--batchnorm', action='store_true', help='Apply batchnorm?')
        subparser.add_argument('--activation', type=str, default='sigmoid', help='Which activation func?')
        subparser.add_argument('--batch-size', type=int, default=32,
                               help='The batch size to use when training the model.')
        subparser.add_argument('--short', action='store_true',
                               help='Use very short datasets to test without loading a bunch of data.')
        subparser.add_argument('--predict-grf-components', type=int, nargs='+', default=[i for i in range(6)],
                               help='Which grf components to train.')
        subparser.add_argument('--predict-cop-components', type=int, nargs='+', default=[i for i in range(6)],
                               help='Which cop components to train.')
        subparser.add_argument('--predict-moment-components', type=int, nargs='+', default=[i for i in range(6)],
                               help='Which moment components to train.')
        subparser.add_argument('--predict-wrench-components', type=int, nargs='+', default=[i for i in range(12)],
                               help='Which wrench components to train.')

    def run(self, args: argparse.Namespace):
        """
        Iterate over all *.b3d files in a directory hierarchy,
        compute file hash, and move to train or dev directories.
        """
        if 'command' in args and args.command != 'visualize':
            return False

        geometry = self.ensure_geometry(args.geometry_folder)

        compare_files = ["Data/d66330dc-7884-4915-9dbb-0520932294c4/MarkerData/SQT01.trc",
                    "LIMO/ComAcc/mot_visualization/latents_subject_run_000cffd9-e154-4ce5-a075-1b4e1fd66201/entry_17_ComAcc.mot", 
                     "LIMO/FinalFinalHigh/mot_visualization/latents_subject_run_d2020b0e-6d41-4759-87f0-5c158f6ab86a/entry_19_FinalFinalHigh.mot"]

        compare_files  = [os.path.join(DATA_DIR, compare_file) for compare_file in compare_files]

        samples = load_samples(compare_files)
        skeletons = [sample.osim.osim.skeleton for sample in samples]

        world = nimble.simulation.World()
        world.setGravity([0, -9.81, 0])

        gui = NimbleGUI(world)
        gui.serve(8000)

        ticker: nimble.realtime.Ticker = nimble.realtime.Ticker(
            0.04)

        frame: int = 0
        playing: bool = True
        num_frames = 196
        if num_frames == 0:
            print('No frames in dataset!')
            exit(1)

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
            # elif key == 'r':
            #     loss_evaluator.print_report()

        gui.nativeAPI().registerKeydownListener(onKeyPress)

        def onTick(now):
            with torch.no_grad():
                nonlocal frame

                # inputs: Dict[str, torch.Tensor]
                # labels: Dict[str, torch.Tensor]
                # inputs, labels, batch_subject_index, trial_index = dev_dataset[frame]
                # batch_subject_indices: List[int] = [batch_subject_index]
                # batch_trial_indices: List[int] = [trial_index]

                # # Add a batch dimension
                # for key in inputs:
                #     inputs[key] = inputs[key].unsqueeze(0)
                # for key in labels:
                #     labels[key] = labels[key].unsqueeze(0)

                # # Forward pass
                # skel_and_contact_bodies = [(dev_dataset.skeletons[i], dev_dataset.skeletons_contact_bodies[i]) for i in batch_subject_indices]
                # outputs = model(inputs)
                # skel = skel_and_contact_bodies[0][0]
                # contact_bodies = skel_and_contact_bodies[0][1]

                # loss_evaluator(inputs, outputs, labels, batch_subject_indices, batch_trial_indices, args, compute_report=True)
                # if frame % 100 == 0:
                #     print('Results on Frame ' + str(frame) + '/' + str(num_frames))
                #     loss_evaluator.print_report(args)

                # # subject_path = train_dataset.subject_paths[batch_subject_indices[0]]
                # # trial_index = batch_trial_indices[0]
                # # print('Subject: ' + subject_path + ', trial: ' + str(trial_index))

                # if output_data_format == 'all_frames':
                #     for key in outputs:
                #         outputs[key] = outputs[key][:, -1, :]
                #     for key in labels:
                #         labels[key] = labels[key][:, -1, :]

                # ground_forces: np.ndarray = outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].numpy()
                # left_foot_force = ground_forces[0, 0:3]
                # right_foot_force = ground_forces[0, 3:6]

                # cops: np.ndarray = outputs[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME].numpy()
                # left_foot_cop = cops[0, 0:3]
                # right_foot_cop = cops[0, 3:6]

                # predicted_forces = (left_foot_force, right_foot_force)
                # predicted_cops = (left_foot_cop, right_foot_cop)

                # pos_in_root_frame = np.copy(inputs[InputDataKeys.POS][0, -1, :].cpu().numpy())
                # pos_in_root_frame[0:6] = 0
                # skel.setPositions(pos_in_root_frame)
                samples[0].osim.osim.skeleton.setPositions(samples[0].osim.motion[frame, :])
                gui.nativeAPI().renderSkeleton(samples[0].osim.osim.skeleton)


                # joint_centers = inputs[InputDataKeys.JOINT_CENTERS_IN_ROOT_FRAME][0, -1, :].cpu().numpy()
                # num_joints = int(len(joint_centers) / 3)
                # for j in range(num_joints):
                #     gui.nativeAPI().createSphere('joint_' + str(j), [0.05, 0.05, 0.05], joint_centers[j * 3:(j + 1) * 3],
                #                                  [1, 0, 0, 1])

                # root_lin_vel = inputs[InputDataKeys.ROOT_LINEAR_VEL_IN_ROOT_FRAME][0, 0, 0:3].cpu().numpy()
                # gui.nativeAPI().createLine('root_lin_vel', [[0, 0, 0], root_lin_vel], [1, 0, 0, 1])

                # root_pos_history = inputs[InputDataKeys.ROOT_POS_HISTORY_IN_ROOT_FRAME][0, 0, :].cpu().numpy()
                # num_history = int(len(root_pos_history) / 3)
                # for h in range(num_history):
                #     gui.nativeAPI().createSphere('root_pos_history_' + str(h), [0.05, 0.05, 0.05],
                #                                  root_pos_history[h * 3:(h + 1) * 3], [0, 1, 0, 1])

                # force_cops = labels[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME][0, :].cpu().numpy()
                # force_fs = labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME][0, :].cpu().numpy()
                # num_forces = int(len(force_cops) / 3)
                # force_index = 0
                # for f in range(num_forces):
                #     if contact_bodies[f] == 'pelvis':
                #         continue
                #     cop = force_cops[f * 3:(f + 1) * 3]
                #     force = force_fs[f * 3:(f + 1) * 3]
                #     gui.nativeAPI().createLine('force_' + str(f),
                #                                [cop,
                #                                 cop + force],
                #                                [1, 0, 0, 1])

                #     predicted_cop = predicted_cops[force_index] # contact_bodies[f].getWorldTransform().translation() #
                #     predicted_force = predicted_forces[force_index]
                #     gui.nativeAPI().createLine('predicted_force_' + str(f),
                #                                [predicted_cop,
                #                                 predicted_cop + predicted_force],
                #                                [0, 0, 1, 1])
                    # force_index += 1

                if playing:
                    frame += 1
                    if frame >= num_frames - 5:
                        frame = 0

        ticker.registerTickListener(onTick)
        ticker.start()
        # Don't immediately exit while we're serving
        gui.blockWhileServing()
        return True


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

def load_samples(compare_files): 
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

    return samples


if __name__ == '__main__':
    logpath = "log"
    # Create and configure logger
    logging.basicConfig(filename=logpath,
                        format='%(asctime)s %(message)s',
                        filemode='a')

    # Creating an object
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    # Setting the threshold of logger to INFO
    logger.setLevel(logging.INFO)



    commands = [VisualizeCommand()]

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description='InferBiomechanics Command Line Interface')

    # Split up by command
    subparsers = parser.add_subparsers(dest="command")

    # Add a parser for each command
    for command in commands:
        command.register_subcommand(subparsers)

    # Parse the arguments
    args = parser.parse_args()

    for command in commands:
        if command.run(args):
            print("Failed in visualization")