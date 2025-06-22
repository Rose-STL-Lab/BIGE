import os
import sys
import copy
import numpy as np 
# Sample loading imports - can be moved to separate file later
dir_path = os.path.dirname(os.path.abspath(__file__))
UCSD_OpenCap_Fitness_Dataset_path = os.path.abspath(os.path.join(dir_path,'..', '..', 'UCSD-OpenCap-Fitness-Dataset' , 'src'))
sys.path.append(UCSD_OpenCap_Fitness_Dataset_path)

from utils import DATA_DIR 
from dataloader import OpenCapDataLoader, MultiviewRGB
from smpl_loader import SMPLRetarget
from osim import OSIMSequence

class WebsiteSample:
    """Class responsible for loading and managing different types of samples"""
    
    def __init__(self, sample_ind,file_path, gui):
        self.sample = WebsiteSample.load_sample(file_path)
        self.selected_dataset = None
        self.selected_experiment = None
        self.selected_trial = None
        self.selected_subject = None
        self.selected_exercise = None
        self.skeleton = None

        self.skeleton = self.sample.osim.osim.skeleton
        self.create_subject_container(sample_ind,gui)       

    
    @staticmethod
    def load_subject(sample_path, retrieval_path=None):
        """Load a subject from a sample path"""
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

    @staticmethod
    def load_retrieved_samples(session, retrieval_path):
        """Load retrieved samples from a session and retrieval path"""
        assert retrieval_path and os.path.isfile(retrieval_path), f"Unable to load .mot file:{retrieval_path}" 

        mot_path = os.path.abspath(retrieval_path)
        print("Loading Generation file:",mot_path)
        
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
        print("Pelvis:",np.rad2deg(sample.osim.motion[::10,1:3]))
        print("KNEE Left:",np.rad2deg(sample.osim.motion[::10,10]))
        print("TIME:",sample.osim.motion[::10,0])
        sample.osim_file = retrieval_path
        return sample

    @staticmethod
    def load_sample(self,file_path):
        """Load samples from a list of file paths"""
        if file_path.endswith('.trc'):
            sample = WebsiteSample.load_subject(file_path)[0]
        elif 'OpenSimData/Dynamics' in file_path: # For Dynamics data
            session = file_path
            for j in range(4):
                session = os.path.dirname(session)
            session = os.path.basename(session)
            sample = WebsiteSample.load_retrieved_samples(session, file_path)
        elif file_path.endswith('.mot'): # For baseline + generated results 
            session = os.path.basename(os.path.dirname(file_path))
            session = session.replace("latents_subject_run_","")
            sample = WebsiteSample.load_retrieved_samples(session, file_path)

        return sample
    

    
    def create_subject_container(self,sample_ind, gui):
        
        
        # def onDropDownChange(values):
        #     print('Drop down changed: ',values)
        #     for sample_ind, sample in enumerate(samples): 
        #         if values == sample.name:
        #             self.sample = samples[sample_ind]

        # ### Create containers and dropdown elements 
        # gui.nativeAPI().createDropDown("Experiment", [sample.name for sample in samples], container_name, onDropDownChange)
        
        container_name = f"Subject-{sample_ind}"
        
        gui.nativeAPI().createCollapsibleContainer(label=container_name, pos=np.array([5, 15]),size=np.array([30, 20]))
        # gui.nativeAPI().createText("LeftTitle", "BIGE Demo", np.array([5, 2]), np.array([100, 6]),layer=container_name)

        # Dropdowns for dataset, experiment, trial, subject, exercise
        gui.nativeAPI().createDropDown("Dataset", 
            ["Dataset1", "Dataset2"], container_name,
            lambda value: self.on_dropdown_change("Dataset", value))
        gui.nativeAPI().createDropDown(
            "Subject-ID", ["Subject1", "Subject2"], container_name,
            lambda value: self.on_dropdown_change("Subject", value),   
        )
        gui.nativeAPI().createDropDown(
            "Exercise", ["Squat"], container_name,
            lambda value: self.on_dropdown_change("Exercise", value),   
        )
        gui.nativeAPI().createDropDown(
            "Method", ["Exp1", "Exp2"], container_name,
            lambda value: self.on_dropdown_change("Experiment", value),   
        )
        gui.nativeAPI().createDropDown(
            "Trial", ["Trial1", "Trial2"], container_name,
            lambda value: self.on_dropdown_change("Trial", value),   
        )
        
        
        return gui
    
    
    def on_dropdown_change(self, key, value):
        print(f"Dropdown changed: {key} -> {value}")
        if key == "Dataset":
            self.selected_dataset = value
            # Update experiment dropdown options based on dataset
            experiments = self.get_experiments_for_dataset(value)
            # self.gui.nativeAPI().setDropdownOptions("Experiment", experiments)
        elif key == "Experiment":
            self.selected_experiment = value
            # Update trial dropdown options based on experiment
            trials = self.get_trials_for_experiment(value)
            # self.gui.nativeAPI().setDropdownOptions("Trial", trials)
        elif key == "Trial":
            self.selected_trial = value
            # Update subject dropdown options based on trial
            subjects = self.get_subjects_for_trial(value)
            # self.gui.nativeAPI().setDropdownOptions("Subject", subjects)
        elif key == "Subject":
            self.selected_subject = value
            # Update exercise dropdown options based on subject
            exercises = self.get_exercises_for_subject(value)
            # self.gui.nativeAPI().setDropdownOptions("Exercise", exercises)
        elif key == "Exercise":
            self.selected_exercise = value
            # Load the selected sample
            self.load_selected_sample()
        else:
            print(f"Unknown dropdown: {key}")

    # Dummy implementations for option retrieval, replace with your logic
    def get_experiments_for_dataset(self, dataset):
        return ["Exp1", "Exp2"]
    def get_trials_for_experiment(self, experiment):
        return ["Trial1", "Trial2"]
    def get_subjects_for_trial(self, trial):
        return ["Subject1", "Subject2"]
    def get_exercises_for_subject(self, subject):
        return ["Squat", "Jump"]

    def load_selected_sample(self):
        print(f"Loading sample for: {self.selected_dataset}, {self.selected_experiment}, {self.selected_trial}, {self.selected_subject}, {self.selected_exercise}")
        # Implement your sample loading logic here