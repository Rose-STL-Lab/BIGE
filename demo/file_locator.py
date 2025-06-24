# This file iterates overs the entire dataset, to query and return sample, names and the path to their MOT files based on query raised by the Dropdown menu in the GUI.
import os
import sys
import tqdm
import pandas as pd


# Sample loading imports - can be moved to separate file later
dir_path = os.path.dirname(os.path.abspath(__file__))
UCSD_OpenCap_Fitness_Dataset_path = os.path.abspath(os.path.join(dir_path,'..', '..', 'UCSD-OpenCap-Fitness-Dataset' , 'src'))
sys.path.append(UCSD_OpenCap_Fitness_Dataset_path)
from utils import DATA_DIR



class FileLocator:
    def __init__(self):
        self.dataset2subjects, self.OpenCapID2PPE = self.load_dataset_to_subjects()
        self.PPE2OpenCapID = {v: k for k, v in self.OpenCapID2PPE.items()}
        self.methods = ['MoCap', 'Simulations', 'BIGE',  'Baselines'] 
        self.experiments = {
                'MoCap': ['OpenCap'], 
                'Simulations': ['Torque Driven', 'Muscle Driven'],
                'BIGE': os.listdir(os.path.join(DATA_DIR, 'LIMO')),
                'Baselines': ['MDM', 'T2M-GPT']
        } 


    def load_dataset_to_subjects(self):
        sub2sess_pd = pd.read_csv(os.path.join(DATA_DIR, 'subject2opencap.txt') ,sep=',')
        OpenCapID2PPE  = dict(zip( sub2sess_pd[' OpenCap-ID'].tolist(),sub2sess_pd['PPE'].tolist()))

        # Test subjects 
        mcs_sessions = ["349e4383-da38-4138-8371-9a5fed63a56a","015b7571-9f0b-4db4-a854-68e57640640d","c613945f-1570-4011-93a4-8c8c6408e2cf","dfda5c67-a512-4ca2-a4b3-6a7e22599732","7562e3c0-dea8-46f8-bc8b-ed9d0f002a77","275561c0-5d50-4675-9df1-733390cd572f","0e10a4e3-a93f-4b4d-9519-d9287d1d74eb","a5e5d4cd-524c-4905-af85-99678e1239c8","dd215900-9827-4ae6-a07d-543b8648b1da","3d1207bf-192b-486a-b509-d11ca90851d7","c28e768f-6e2b-4726-8919-c05b0af61e4a","fb6e8f87-a1cc-48b4-8217-4e8b160602bf","e6b10bbf-4e00-4ac0-aade-68bc1447de3e","d66330dc-7884-4915-9dbb-0520932294c4","0d9e84e9-57a4-4534-aee2-0d0e8d1e7c45","2345d831-6038-412e-84a9-971bc04da597","0a959024-3371-478a-96da-bf17b1da15a9","ef656fe8-27e7-428a-84a9-deb868da053d","c08f1d89-c843-4878-8406-b6f9798a558e","d2020b0e-6d41-4759-87f0-5c158f6ab86a","8dc21218-8338-4fd4-8164-f6f122dc33d9"]
        skip_subjects = []
        sessions = os.listdir(os.path.join(DATA_DIR, 'Data'))
        
        dataset2subjects = {'Training': [], 'Testing(MCS Subjects)': []}
        
        
        for subject_ind, subject_name in tqdm.tqdm(enumerate(sessions)):
            # print(f"Checking Subject:{subject_ind} Name:{subject_name}")

            if subject_name in skip_subjects: continue
            if not os.path.isdir(os.path.join(DATA_DIR, 'Data',  subject_name)): continue
            if subject_name not in OpenCapID2PPE: continue
            
            if subject_name in mcs_sessions:
                dataset2subjects['Testing(MCS Subjects)'].append(OpenCapID2PPE[subject_name])
            else:
                dataset2subjects['Training'].append(OpenCapID2PPE[subject_name])
        return dataset2subjects, OpenCapID2PPE
    
    
    def get_trials(self, subject, method, experiment=None):
        """
            Get valid trials for a given subject, method and experiment.
        Args:
            subject (str): Subjets in OpenCap, 
            method (str):  Method to use, one of ['MoCap', 'Simulations', 'BIGE', 'Baselines']
            experiment (str): Specific experiment to filter trials.
            
        Returns:
            list: List of trials for the given subject, method and experiment.
            error(str): Error message if subject not found or method/experiment not recognized.
        """ 
        if subject not in self.PPE2OpenCapID:

            return [], [], f"Subject {subject} not found in PPE2OpenCapID mapping."
        
        subject = self.PPE2OpenCapID[subject]
        
        trials = []
        trials_path = []
        error = ""
        
        
        if method == 'MoCap':
            if experiment is None:
                experiment = 'OpenCap'  # Default to OpenCap if no experiment is specified
            
                
            if experiment == 'OpenCap':
                trials = os.listdir(os.path.join(DATA_DIR, 'Data', subject, 'OpenSimData', 'Kinematics'))
                trials = [trial.replace('.mot', '') for trial in trials if trial.endswith('.mot')]
                trials_path = [os.path.join(DATA_DIR, 'Data', subject, 'OpenSimData', 'Kinematics', trial + '.mot') for trial in trials]
            else:
                error = f"Experiment {experiment} not recognized for MoCap method."
        elif method == 'Simulations':
            if experiment is None:
                experiment = 'Muscle Driven'  # Default to Muscle Driven if no experiment is specified 
            
            
            dynamics_path = os.path.join(DATA_DIR, 'Data', subject, 'OpenSimData', 'Dynamics') 
            for trial in os.listdir(dynamics_path): 
                if os.path.isdir(os.path.join(dynamics_path, trial)):
                    for file in os.listdir(os.path.join(dynamics_path, trial)):
                        if 'kinematics' in file and 'torque_driven.mot' in file and file.endswith('.mot')  and experiment == 'Torque Driven':
                            trials.append(trial)
                            trials_path.append(os.path.join(dynamics_path, trial, file))
                        elif 'kinematics' in file and 'muscle_driven.mot' in file and file.endswith('.mot') and experiment == 'Muscle Driven':
                            trials.append(trial)
                            trials_path.append(os.path.join(dynamics_path, trial, file))
                            
            if len(trials) == 0:
                error =  f"No trials found for {experiment} in Simulations method for subject {subject}."

        
        elif method == 'BIGE':
            
            if experiment is None:
                experiment = 'FinalFinalHigh'  # Default to Muscle Driven if no experiment is specified
            
            
            
            if experiment in self.experiments['BIGE']:
                trials = os.listdir(os.path.join(DATA_DIR, 'LIMO', experiment, 'mot_visualization', f'latents_subject_run_{subject}'))
                trials = [trial.replace(f'_{experiment}.mot', '') for trial in trials if trial.endswith('.mot')]
                trials_path = [os.path.join(DATA_DIR, 'LIMO', experiment, 'mot_visualization', f'latents_subject_run_{subject}', trial + f'_{experiment}.mot') for trial in trials]

            else:
                error = f"Experiment {experiment} not recognized for BIGE method."
        
        elif  method == 'Baselines':
            if experiment == 'MDM':
                trials = os.listdir(os.path.join(DATA_DIR, 'baslines', 'm2m_basline', subject))
                trials = [trial.replace(f'.mot', '') for trial in trials if trial.endswith('.mot')]
                trials_path = [os.path.join(DATA_DIR, 'baslines', 'm2m_basline', subject, trial + '.mot') for trial in trials] 
            elif experiment == 'T2M-GPT':
                trials = os.listdir(os.path.join(DATA_DIR, 't2m_baseline', subject))
                trials = [trial.replace(f'.mot', '') for trial in trials if trial.endswith('.mot')]
                trials_path = [os.path.join(DATA_DIR, 't2m_baseline', subject, trial + '.mot') for trial in trials]
            else:
                error = f"Experiment {experiment} not recognized for Baselines method."
        else:
            error = f"Method {method} not recognized."

        assert len(trials) == len(trials_path), "Mismatch between trials and trials_path lengths."

        if len(trials) == 0:
            error =  f"No trials found for {experiment} in {method} method for subject {subject}."


        return trials, trials_path, error
    
    def get_sample_info_from_path(self, file_path):
        """
        Extract sample information from a file path.
        Args:
            file_path (str): Path to the file.

        Returns:
            file_info: (subject, method, experiment, trial)
            error (str): Error message if the file path structure is unrecognized.
        """
        file_path = os.path.abspath(file_path)
        file_path = file_path.replace(os.path.abspath(DATA_DIR)+'/', '')
        
        dataset = ''
        subject = ''
        method = ''
        experiment = ''
        trial = ''
        error = ''
        
        parts = file_path.split(os.sep)
        if parts[0] == 'Data':
            if 'OpenSimData' in parts and 'Kinematics' in parts:
                subject = parts[1]
                method = 'MoCap'
                experiment = 'OpenCap'
                trial = parts[-1].replace('.mot', '')
            elif 'MarkerData' in parts:
                subject = parts[1]
                method = 'MoCap'
                experiment = 'OpenCap'
                trial = parts[-1].replace('.trc', '')
            
            elif 'OpenSimData' in parts and 'Dynamics' in parts:
                subject = parts[1]
                method = 'Simulations'
                experiment = 'Torque Driven' if 'torque_driven' in parts[-1] else ''
                experiment = 'Muscle Driven' if 'muscle_driven' in parts[-1] else ''
                if experiment == 'Torque Driven' or experiment == 'Muscle Driven':
                    trial = parts[-2]
            else: 
                error = f"Unrecognized file path structure for MoCap or Simulations: {file_path}"
        elif parts[0] == 'LIMO':
            if len(parts) < 5:
                error = f"Unrecognized file path structure for BIGE: {file_path}"
            else: 
                method = 'BIGE'
                experiment = parts[1]
                subject = parts[3].replace('latents_subject_run_', '')
                trial = parts[-1].replace('.mot', '')
        elif 'baseline' in parts[0]:
            method = 'Baselines'
            if 'm2m_baseline' in parts:
                experiment = 'MDM'
                subject = parts[-2]      
                trial = parts[-1].replace('.mot', '')
            elif 't2m_baseline' in parts:
                experiment = 'T2M-GPT'
                subject = parts[-2]
                trial = parts[-1].replace('.mot', '')
            else: 
                error = f"Unrecognized baseline experiment in file path: {file_path}"
        else:
            error = f"Unrecognized file path structure: {file_path}"

        if subject in self.OpenCapID2PPE:        
            subject = self.OpenCapID2PPE[subject]
        else:
            error = f"Subject {subject} not found in PPE2OpenCapID mapping."
        
        if subject in self.dataset2subjects['Testing(MCS Subjects)']:
            dataset = 'Testing(MCS Subjects)'
        elif subject in self.dataset2subjects['Training']:
            dataset = 'Training'
        else:
            error = f"Subject {subject} not found in dataset mapping."

        if not dataset or not subject or not method or not experiment or not trial:
            error = f"Unable to extract sample information from file path: {file_path}, dataset: {dataset}, subject: {subject}, method: {method}, experiment: {experiment}, trial: {trial}"
            return None, error
        
        return (dataset, subject, method, experiment, trial), error

if __name__ == "__main__":
    locator = FileLocator()