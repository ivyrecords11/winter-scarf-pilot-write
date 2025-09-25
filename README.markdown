# DUE TO : 25/09/??
- Design/Evaluate learning of various SNN RL models
  
# Libraries Used
- PyTorch
- SpikingJelly https://github.com/fangwei123456/spikingjelly
- MuJoCo Physical Simulator https://mujoco.org/

# Directories

## folders
- .venv : 
  Virtual Environment for running code
  
- old : 
  stack of old codes
  
- pytorch : 
  build for using xpu (Intel GPU of personal laptop)
  
- spikingjelly : 
  module file for SpikingJelly Library

## files

- config.py :
  Dataclass of simulation variables
  - SimulationConfig

- mujoco_model.py : 
  Simulation environment built with mujoco

- Environment(cfg:SimulationConfig)

### SNNModels.py
- Various models/config of SNN

### viewer_utils.py
- Launches mujoco.viewer for simulation visualization

- simpleDeepRL_* : 
  Reinforced Learning with Simple ANN

### tiltPlateNoSensor.xml
you can load the file into the installable Mujoco simulator program and see how it looks


# UPDATES
### TODO 25/09/23
- ~mujoco 환경 구현~
- ~간단한 DNN 모델 강화학습~

### TODO 25/09/24
- ~모터 속도 = 스파이크 이동평균에 비례하게 변경~
- sensor delay 반영
- ~reward 초당 값으로 조정~
- SNN 모델 구조 변경 (CerebellarNet, CerebellarCNN)
  - Deep SNN
  - ~CerebellarNet~
  - CerebellarCNN
  - CerebellarCF
- SNN 모델 학습 코드 작성 (run_rl.py)

### TODO ??
- config 초 단위로 조정
- optuna로 SNN 파라미터 설정
- RL QAT하는 법 알아보기












