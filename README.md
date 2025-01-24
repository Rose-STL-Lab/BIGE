# BIGE: Biomechanics-informed GenAI for Exercise Science 
 
[![](https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue)](https://rose-stl-lab.github.io/UCSD-OpenCap-Fitness-Dataset/)
[![](https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green)](https://rose-stl-lab.github.io/UCSD-OpenCap-Fitness-Dataset/static/pdfs/L4DC_2025_paper_177.pdf)
[![](https://img.shields.io/badge/Code-Github-red?style=flat&logo=github)](https://gitlab.nrp-nautilus.io/shmaheshwari/digital-coach-anwesh.git)
[![](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/your-repo/actions)
<!-- ![](https://img.shields.io/badge/Windows-0078D6?style=flat&logo=windows&logoColor=white) -->
![](https://img.shields.io/badge/Ubuntu20.04-E95420?style=flat&logo=ubuntu&logoColor=white)



![Teaser figure](https://rose-stl-lab.github.io/UCSD-OpenCap-Fitness-Dataset/static/images/Sports_Analytic_Mockup_1_V4.png)

> **BIGE** is a framework for generative models to adhere to clinician-defined constraints.  To generate realistic motion, our method uses a biomechanically informed surrogate model to guide the generation process.

## Summary

Proper movements enhance mobility, coordination, and muscle activation, which are crucial for performance, injury prevention, and overall fitness. However, traditional simulation tools rely on strong modeling assumptions, are difficult to set up, and are computationally expensive. On the other hand, generative AI approaches provide efficient alternatives to motion generation. However they often lack physiological relevance and do not incorporate biomechanical constraints, limiting their practical applications in sports and exercise science. To address these limitations:

- We propose a novel framework, BIGE, that combines bio-mechanically meaningful scoring metrics with generative modeling.
- BIGE integrates a differentiable surrogate model for muscle activation to reverse optimize the latent space of the generative model.
- Enables the retrieval of physiologically valid motions through targeted search.
- Through extensive experiments on squat exercise data, our framework demonstrates superior performance in generating diverse, physically plausible motions while maintaining high fidelity to clinician-defined objectives compared to existing approaches.


## Table of Content
* [1. Visual Results](#1-visual-results)
* [2. Installation](#2-installation)
* [3. Quick Start](#3-quick-start)
* [4. Train](#4-train)
* [5. Evaluation](#5-evaluation)
* [6. SMPL Mesh Rendering](#6-smpl-mesh-rendering)
* [7. Acknowledgement](#7-acknowledgement)
* [8. Commands](#Commands)






## 1. Visual Results (More results can be found on our [project page](https://rose-stl-lab.github.io/UCSD-OpenCap-Fitness-Dataset))
<img src="https://rose-stl-lab.github.io/UCSD-OpenCap-Fitness-Dataset/static/images/BIGE_Qualitative_Diagram_V4.jpeg" alt="alt text" style="max-height: 960px;">

> Our guidance strategy leads to a more physiologically accurate squat motion as evidenced by the increased squat depth. The generated motion samples are ordered by the peak muscle activation. The red and green lines at 50% squat cycle represent the depth of the squat. 


<video controls style="max-height: 960px;">
  <source src="https://rose-stl-lab.github.io/UCSD-OpenCap-Fitness-Dataset/static/videos/comparision.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
 
> Comparison of generated samples from baselines and BIGE. The yellow curve represents the movement of the hip joint over the entire squat cycle. BIGE generates a more realistic squat motion compared to baselines.

## 2. Installation

### 2.1. Environment


Our model can be learned in a **single GPU V100-32G**

```bash
conda env create -f environment.yml
conda activate T2M-GPT

ulimit -n 1000000 # Need to run pytorch open multiple files https://stackoverflow.com/questions/71642653/how-to-resolve-the-error-runtimeerror-received-0-items-of-ancdata
```

The code was tested on Python 3.8 and PyTorch 1.8.1.

### 2.3. Datasets

More details about the dataset used can be found here [[here]](https://rose-stl-lab.github.io/UCSD-OpenCap-Fitness-Dataset/ucsd-opencap-dataset.html)

### 2.5. Pre-trained models 

The pretrained model files will be stored in the 'pretrained' folder:

## 3. DEMO (Coming Soon)


## 4. VQ-VAE

Train the VQVAE model with specified parameters.
```bash
python3 train_vq.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir output --dataname mcs --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name VQVAE9
```
<details>
<summary>VQVAE Training without DeepSpeed</summary>
Train the VQVAE model using DeepSpeed for optimized performance.

```bash
python3.8 /home/ubuntu/.local/bin/deepspeed train_vq.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir output --dataname mcs --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name VQVAE9
```

</details>

<details> <summary>VQVAE Reconstruction</summary>

Generate samples using the trained VQVAE model.

```bash
python MOT_eval.py --dataname mcs --out-dir output --exp-name VQVAE5_v2 --resume-pth output/VQVAE5_v2/300000.pth
```

</details>


## 5. Surrogate
Run surrogate training for the model.
```bash
python3.8 surrogate_training.py
```

## 6. Guidance

```bash
python LIMO_Surrogate.py --exp-name TestBIGE --vq-name /home/ubuntu/data/T2M-GPT/output/VQVAE14/120000.pth  --dataname mcs --seq-len 49 --total-iter 3000 --lr 0.5 --num-runs 3000 --min-samples 20  --subject /data/panini/MCS_DATA/Data/d66330dc-7884-4915-9dbb-0520932294c4 --low 0.35 --high 0.45
```

<details><summary>Run guidance for all subjects</summary>

```python
import os
mcs_sessions = ["349e4383-da38-4138-8371-9a5fed63a56a","015b7571-9f0b-4db4-a854-68e57640640d","c613945f-1570-4011-93a4-8c8c6408e2cf","dfda5c67-a512-4ca2-a4b3-6a7e22599732","7562e3c0-dea8-46f8-bc8b-ed9d0f002a77","275561c0-5d50-4675-9df1-733390cd572f","0e10a4e3-a93f-4b4d-9519-d9287d1d74eb","a5e5d4cd-524c-4905-af85-99678e1239c8","dd215900-9827-4ae6-a07d-543b8648b1da","3d1207bf-192b-486a-b509-d11ca90851d7","c28e768f-6e2b-4726-8919-c05b0af61e4a","fb6e8f87-a1cc-48b4-8217-4e8b160602bf","e6b10bbf-4e00-4ac0-aade-68bc1447de3e","d66330dc-7884-4915-9dbb-0520932294c4","0d9e84e9-57a4-4534-aee2-0d0e8d1e7c45","2345d831-6038-412e-84a9-971bc04da597","0a959024-3371-478a-96da-bf17b1da15a9","ef656fe8-27e7-428a-84a9-deb868da053d","c08f1d89-c843-4878-8406-b6f9798a558e","d2020b0e-6d41-4759-87f0-5c158f6ab86a","8dc21218-8338-4fd4-8164-f6f122dc33d9"]
exp_name = "FinalFinalHigh"
for session in mcs_sessions:
  os.system(f"python LIMO_Surrogate.py --exp-name {exp_name} --vq-name /data/panini/T2M-GPT/output/VQVAE14/120000.pth  --dataname mcs --seq-len 49 --total-iter 3000 --lr 0.5 --num-runs 3000 --min-samples 20  --subject /data/panini/MCS_DATA/Data/{session} --low 0.35 --high 0.45")
```

</details>



## 7. Metrics 
### 7.1 Calculate Wasserstein Distance & Entropy


  - For BIGE
    ```
    python calculate_wasserstein.py --file_type mot --folder_path /home/ubuntu/data/MCS_DATA/LIMO/FinalFinalHigh/mot_visualization/
    ``` 

 - For motion capture
    ```bash
    python wasserstein_mocap.py --file_type mot --folder_path /home/ubuntu/data/MCS_DATA/Data/
    ```
  - For baselines 
    ```bash
    python calculate_wasserstein.py --file_type mot --folder_path /home/ubuntu/data/MCS_DATA/baselines/mdm_baseline/
    python calculate_wasserstein.py --file_type mot --folder_path /home/ubuntu/data/MCS_DATA/baselines/t2m_baseline/
    python calculate_wasserstein.py --file_type mot --folder_path /home/ubuntu/data/MCS_DATA/LIMO/VQVAE-Generations/mot_visualization/
    ```


### 7.2 Calculate Guidance metrics

  - For references 
    ```bash
    python calculate_guidance.py --file_type mocap --folder_path /home/ubuntu/data/MCS_DATA/Data/ 
    ```

  - For baselines 
    ```
      python calculate_guidance.py  --file_type mot --folder_path /home/ubuntu/data/MCS_DATA/baselines/mdm_baseline/
      python calculate_guidance.py --file_type mot --folder_path /home/ubuntu/data/MCS_DATA/baselines/t2m_baseline/
      python calculate_guidance.py --file_type mot --folder_path /home/ubuntu/data/MCS_DATA/LIMO/VQVAE-Generations/mot_visualization/
    ``` 

  - For BIGE 
    ```   
    python calculate_guidance.py --file_type mot --folder_path /home/ubuntu/data/MCS_DATA/LIMO/FinalFinalHigh/mot_visualization/
    ```

## 8. Rendering

To render videos and images, install the [UCSD-OpenCap-Fitness-Dataset](https://rose-stl-lab.github.io/UCSD-OpenCap-Fitness-Dataset/ucsd-opencap-dataset.html)  

### Installation 

```
git clone 
```


Generate MP4 videos from MOT files.
```bash
cd UCSD-OpenCap-Fitness_Dataset/
export DISPLAY=:99.0
```


### Compare MoCap and generated sample

```
python src/opencap_reconstruction_render.py <absolute subject-path> <absolute mot-path> <absolute save-path> # Compare 
```


### Compare 3 .mot files 

```
python src/plot_3.py <mot-path-1> <mot-path-2> <mot-path-3> <optonal-video-path>
```

<details><summary>Examples: </summary>   
- BIGE
```
python src/plot_3.py  MCS_DATA/LIMO/FinalFinalHigh/mot_visualization/latents_subject_run_d66330dc-7884-4915-9dbb-0520932294c4/entry_{0,2,19}_FinalFinalHigh.mot render/bige_3.mp4
```

- MDM  
```
python src/plot_3.py  MCS_DATA/mdm_baseline/015b7571-9f0b-4db4-a854-68e57640640d/results_*_radians.mot
```


- T2M
```
python src/plot_3.py  MCS_DATA/mdm_baseline/015b7571-9f0b-4db4-a854-68e57640640d/results_*_radians.mot
```


- Simulation 
```
 python src/plot_3.py  MCS_DATA/Data/c613945f-1570-4011-93a4-8c8c6408e2cf/OpenSimData/Dynamics/SQT01_segment_?/kinematics_activations_SQT01_segment_?_muscle_driven.mot
```


- For MoCap data collected using opencap
```
 python src/plot_3.py MCS_DATA/Data/fb6e8f87-a1cc-48b4-8217-4e8b160602bf/MarkerData/SQT01.trc
```


- For Simulation data
```
python src/plot_3.py  MCS_DATA/Data/<subject-path>/OpenSimData/Dynamics/SQT01_segment_?/kinematics_activations_SQT01_segment_?_muscle_driven.mot 

```

- OpenCap Dataset 



</details>




### 9. Acknowledgment

We extend our gratitude to The Wu Tsai Human Performance Alliance for their invaluable support and resources. Their dedication to advancing human performance through interdisciplinary research has been instrumental to our project.

This work utilized resources provided by the National Research Platform (NRP) at the University of California, San Diego. The NRP is developed and supported in part by funding from the National Science Foundation under awards 1730158, 1540112, 1541349, 1826967, 2112167, 2100237, and 2120019, along with additional support from community partners.

We also acknowledge the contributions of public code repositories such as [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [MDM](https://github.com/GuyTevet/motion-diffusion-model), and [MotionDiffuse](https://github.com/mingyuan-zhang/MotionDiffuse).

Additionally, we appreciate the open-source motion capture systems like [OpenCap](https://github.com/stanfordnmbl/opencap-processing).

