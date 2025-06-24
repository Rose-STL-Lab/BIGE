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
from osim import OSIMSequence, load_mot, load_osim

class WebsiteSample:
    """Class responsible for loading and managing different types of samples"""
    
    def __init__(self, sample_ind,file_path, gui, file_locator):
        

        self.file_locator = file_locator
        self.gui = gui
        
        file_info, error = file_locator.get_sample_info_from_path(file_path)
        if error:
            print(f"File locator unable to find the sample: {error}")
            return 
        else: 
            self.selected_dataset, self.selected_subject, self.selected_method, self.selected_experiment, self.selected_trial = file_info
            trials, trial_path, error = self.file_locator.get_trials(self.selected_subject, self.selected_method, experiment=self.selected_experiment)
            if error:
                print(f"File locator unable to find the trials: {error}")
                return
            self.trials = trials 
            self.trial_path = trial_path
            print(self.trial_path[1]) 
        self.container_name = self.create_subject_container(sample_ind,gui)       
        
        self.sample = WebsiteSample.load_sample(file_path)

        self.skeleton = self.sample.osim.osim.skeleton        

    
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
    def load_sample(file_path):
        """Load samples from a list of file paths"""
        if file_path.endswith('.trc'):
            sample = WebsiteSample.load_subject(file_path)[0]
        elif 'OpenSimData/Kinematics' in file_path: # For Kinematics data
            session = file_path
            for j in range(3):
                session = os.path.dirname(session)
            session = os.path.basename(session)
            sample = WebsiteSample.load_retrieved_samples(session, file_path)
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
        
        container_name = f"{self.selected_method}"
        
        gui.nativeAPI().createCollapsibleContainer(label=container_name, pos=np.array([5, 15]),size=np.array([30, 20]))
        # gui.nativeAPI().createText("LeftTitle", "BIGE Demo", np.array([5, 2]), np.array([100, 6]),layer=container_name)

        # Dropdowns for dataset, experiment, trial, subject, exercise
        gui.nativeAPI().createDropDown("Dataset", 
            self.file_locator.dataset2subjects.keys(), container_name,
            lambda value: self.on_dropdown_change("Dataset", value))
        gui.nativeAPI().setDropDownValue("Dataset", container_name, self.selected_dataset)
        
        gui.nativeAPI().createDropDown(
            "Subject", self.file_locator.dataset2subjects[self.selected_dataset], container_name,
            lambda value: self.on_dropdown_change("Subject", value),   
        )
        gui.nativeAPI().setDropDownValue("Subject", container_name, self.selected_subject)
        
        gui.nativeAPI().createDropDown(
            "Method", self.file_locator.methods, container_name,
            lambda value: self.on_dropdown_change("Method", value),   
        )
        gui.nativeAPI().setDropDownValue("Method", container_name, self.selected_method)
        
        gui.nativeAPI().createDropDown(
            "Experiment", self.file_locator.experiments[self.selected_method], container_name,
            lambda value: self.on_dropdown_change("Experiment", value),   
        )
        gui.nativeAPI().setDropDownValue("Experiment", container_name, self.selected_experiment)
        
        
        gui.nativeAPI().createDropDown(
            "Trial", self.trials, container_name,
            lambda value: self.on_dropdown_change("Trial", value),   
        )
        gui.nativeAPI().setDropDownValue("Trial", container_name, self.selected_trial)
        
        # import time  
        # time.sleep(10)  # Wait for GUI to initialize properly
        # gui.nativeAPI().setDropDownOptions("Experiment", container_name, self.file_locator.experiments["Baselines"])
        
        return container_name
    
    def set_trials(self):
        """Set trials and their paths for the current sample"""
        trials, trials_path, error = self.file_locator.get_trials(self.selected_subject, self.selected_method, experiment=self.selected_experiment)
        if error: 
            print(f"Error retrieving trials for subject {self.selected_subject}: {error}")
            return False
        self.trials, self.trial_path = trials, trials_path
        print(self.trials, self.trial_path)
        self.gui.nativeAPI().setDropDownOptions("Trial", self.container_name, trials)
        return True
    def load_subject_osim(self):
        """Load the subject' skeletons
           @returns: True if the OSIM file was loaded successfully, False otherwise
        """
        subject_opencap_id = self.file_locator.PPE2OpenCapID.get(self.selected_subject, None)
        if subject_opencap_id is None:
            print(f"OpenCap ID not found for subject {self.selected_subject}. Please check the file locator.")
            return False
        osim_path = os.path.join(DATA_DIR, 'Data', subject_opencap_id, 'OpenSimData', 'Model', 'LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim') 
        if not os.path.exists(osim_path):
            for possible_path in os.listdir(os.path.join(DATA_DIR, 'Data', subject_opencap_id, 'OpenSimData', 'Model')):
                if possible_path.endswith('.osim'):
                    osim_path = os.path.join(DATA_DIR, 'Data', subject_opencap_id, 'OpenSimData', 'Model', possible_path)
                    break
        if not os.path.exists(osim_path):
            print(f"OSIM file not found for subject {self.selected_subject} at {osim_path}. Using default model.")
            return False

        osim_geometry_path = os.path.join(DATA_DIR,'OpenCap_LaiArnoldModified2017_Geometry')
        
        self.sample.osim.osim = load_osim(osim_path, geometry_path=osim_geometry_path)
    
    
    def on_dropdown_change(self, key, value):
        print(f"Dropdown changed: {key} -> {value}")
        try:

            if key == "Dataset":
                self.selected_dataset = value
                subjects = self.file_locator.dataset2subjects.get(self.selected_dataset, [])
                print(f"Subjects for dataset {self.selected_dataset}: {subjects}")
                self.gui.nativeAPI().setDropDownOptions("Subject", self.container_name, subjects)
                if "train" in self.selected_dataset.lower():
                    self.selected_method = "MoCap"
                    self.gui.nativeAPI().setDropDownValue("Method", self.container_name, self.selected_method)
                    self.gui.nativeAPI().setDropDownOptions("Experiment", self.container_name, self.file_locator.experiments[self.selected_method])
                    self.selected_experiment = "OpenCap"
                    self.gui.nativeAPI().setDropDownValue("Experiment", self.container_name, self.selected_experiment)

                    if not self.set_trials():
                        print(f"No trials found for dataset {self.selected_dataset} method {self.selected_method} experiement {self.selected_experiment}")
                        self.gui.nativeAPI().setDropDownOptions("Trial", self.container_name, [])
                        return

                else:
                    self.selected_method = "BIGE"
                    self.gui.nativeAPI().setDropDownValue("Method", self.container_name, self.selected_method)
                    self.gui.nativeAPI().setDropDownOptions("Experiment", self.container_name, self.file_locator.experiments[self.selected_method])
                    self.selected_experiment = "FinalFinalHigh"
                    self.gui.nativeAPI().setDropDownValue("Experiment", self.container_name, self.selected_experiment)
                    
                    if not self.set_trials():
                        print(f"No trials found for dataset {self.selected_dataset}")
                        self.gui.nativeAPI().setDropDownOptions("Trial", self.container_name, [])
                        return
            elif key == "Subject":                
                value_id = self.file_locator.dataset2subjects[self.selected_dataset].index(value)
                if value_id < 0:
                    raise KeyError(f"Invalid subject selection: {value}")
                self.selected_subject = value                
                # Update trials dropdown options based on subject
                if not self.set_trials():
                    print(f"No trials found for dataset {self.selected_dataset}")
                    self.gui.nativeAPI().setDropDownOptions("Trial", self.container_name, [])
                    return
                self.load_subject_osim()                

            elif key == "Method":
                print(f"Selected method: {value}")
                value_id = self.file_locator.methods.index(value)
                if value_id < 0:
                    print(f"Invalid method selection: {value}")
                    return
                else: 
                    print(f"Selected method ID: {value_id}")
                    self.selected_method = value                
                    
                
                self.gui.nativeAPI().setDropDownOptions("Experiment", self.container_name, self.file_locator.experiments[self.selected_method])
                self.selected_experiment = None
                for experiment in self.file_locator.experiments[self.selected_method]:
                    self.selected_experiment = experiment 
                    if not self.set_trials():
                        self.selected_experiment = None
                        continue  # If trials could not be set, skip to next experiment
                    else: 
                        print(f"Found trials for method {self.selected_method} and experiment {self.selected_experiment}")
                        break  # Break after finding the first valid experiment with trials
                
                if self.selected_experiment is None:
                    print(f"No valid experiment found for method {self.selected_method}")
                    return
                else: 
                    print(f"Selected method: {self.selected_method} for subject: {self.selected_subject}")
                    self.gui.nativeAPI().setDropDownValue("Experiment", self.container_name, self.selected_experiment)
            
            elif key == "Experiment":
                value_id = self.file_locator.experiments[self.selected_method].index(value)
                if value_id < 0:
                    print(f"Invalid experiment selection: {value}")
                    return 
                    
        
                self.selected_experiment = value
                # Update trial dropdown options based on experiment
                if not self.set_trials():
                    print(f"No trials found for experiment {self.selected_experiment} in method {self.selected_method}")
                    return
                else:
                    print(f"Selected experiment: {self.selected_experiment} for subject: {self.selected_subject}")
            elif key == "Trial":
                value_id = self.trials.index(value)
                if value_id < 0:
                    print(f"Invalid trial selection: {value}")
                    
                
                self.selected_trial = value
                trial_path = self.trial_path[value_id]
                motion = load_mot(self.sample.osim.osim,  trial_path)
                print(f"Loaded motion data for trial: {self.selected_trial} from {trial_path} {motion.shape}")
                self.sample.osim.motion = motion        
    
            else:
                print(f"Unknown dropdown: {key}")
        except Exception as e:
            print(f"Error loading motion data:{e}")

