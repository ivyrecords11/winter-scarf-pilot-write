#@title Imports and global settings
from dataclasses import dataclass
import mujoco as mj
import numpy as np
import os
import torch
from torch import nn
import spikingjelly
from spikingjelly.activation_based import layer, neuron, functional, learning, surrogate, encoding
from math import ceil

"""
TODO 25/09/23
- mujoco 환경 구현
- 간단한 DNN 모델 강화학습
TODO 25/09/24
- sensor delay 반영
- reward 초당 값으로 조정
- SNN 모델 구조 변경 (CerebellarNet, CerebellarCNN)
- SNN 모델 학습 코드 작성 (run_rl.py)
- 모터 속도 = 스파이크 이동평균에 비례
TODO
- optuna로 SNN 파라미터 설정
- RL QAT하는 법 알아보기
"""

@dataclass
class SimulationConfig:
    plate_size: float = 0.3 # (m)
    n_mf : int = 100       # 모형화된 모세포 수
    n_grc: int = 1024     # 모형화된 과립세포 수
    n_pkj: int = 32        # 모형화된 푸르키네 세포 수
    n_cf: int = 8         # 모형화된 교세포 수
    n_motor: int = 4      # 제어할 모터 수

    ball_mass: float = None    # 공 질량 (kg), None이면 리셋 시 무작위
    tau_pre_ms: float = 5.0    # STDP 전-시냅스 가중치 변화 시간상수 (ms)
    tau_post_ms: float = 5.0   # STDP 후-시냅스 가중치 변화 시간상수 (ms)
    tau_e_ms: float = 50.0     # 시냅스 효율 시간상수 (ms)
    dt: float = 1e-3           # 시뮬레이션 시간 간격 (ms)
    eta: float = 1e-3          # 학습률
    seed: int = None            # 난수 시드
    # sensor parameters
    max_firing_rate: float = (1.0)/dt  # 센서 최대 발화율 (Hz)
    # training parameters
    # every reward scale is per second 1000*dt
    simulation_duration_s: float = 10.0  # 각 에피소드 시뮬레이션 시간 (초)
    batch_size: int = 16          # 배치 크기
    success_timestep: int = 3/dt  # 성공으로 간주할 연속 스텝 수 (3초)
    success_reward: float = 10 # 성공시 보너스(timestep당)(평균 success_frame*4)
    penalize_failure: float = 10.0
    # physic parameters
    ball_mass: float = None  # 공 질량 (kg), None이면 리셋 시 무작위
    max_delay_steps: int = 50 # 최대 지연 스텝 수 (50m/s: dt=0.1ms 기준 5ms * 0.15m/1m)
    #    max_hinge_deg: float = 5.0  # 판 최대 경사각 (도)

class DelayLine:
    def __init__(self, max_steps, n_channels):
        self.buf = np.zeros((max_steps+1, n_channels), dtype=np.float32)
        self.ptr = 0
        self.max_steps = max_steps
        self.n = n_channels
    def push(self, x_t):  # x_t shape [n_channels]
        self.buf[self.ptr, :] = x_t
        self.ptr = (self.ptr + 1) % (self.max_steps+1)
    def read_delayed(self, ks):  # ks shape [n_channels], per-channel delay steps
        # gather diagonal indices with wrap-around
        idx = (self.ptr - 1 - ks) % (self.max_steps+1)
        return self.buf[idx, np.arange(self.n)]

class Environment:
    def __init__(self, cfg: SimulationConfig):
        # config
        self.cfg = cfg
        self.plate_size = cfg.plate_size
        self.n_mf = cfg.n_mf
        self.dt = cfg.dt  # 초 단위로 변환
        # ball params
        self.ball_mass = cfg.ball_mass #(kg)
        self.ball_radius = self.ball_mass ** (1/3) / 10 if self.ball_mass else None #(m)
        self.ball_x : float
        self.ball_y : float
        self.ball_mass_n : float = (self.ball_mass - 0.001) / 0.029 if self.ball_mass else None  # [0,1]로 정규화
        #sensor params
        self.max_firing_rate = 200.0  # Hz
        # Simulation params
        self.success_timestep = cfg.success_timestep
        self.success_count : int = 0
        self.success_reward : float = cfg.success_reward  # 성공시 보너스
        self.success_reward_bonus : float = self.success_reward/self.dt  # 성공시 보너스
        self.failure_reward_penalty : float = cfg.penalize_failure*cfg.simulation_duration_s/cfg.dt # 실패시 패널티
        self.simulation_duration_s : float = cfg.simulation_duration_s
        self.step_count : int = 0
        #self.seed = cfg.seed
        # local params
        self.model: mj.MjModel
        self.data: mj.MjData
        self.sensor_pos_grid = np.linspace(-self.plate_size*0.45, self.plate_size*0.45, 10)
        
        # ... (환경 초기화 코드)

    def sensor_inputs(self, flatten=True):
        """
        OUTPUTS
            inputs: (10,10) spike encoded sensor inputs (torch.float32)
        """
        n=10
        p = torch.zeros((n, n), dtype=torch.float32)
        for i in range(n):
            for j in range(n):
                dist = np.sqrt((self.ball_x - self.sensor_pos_grid[i])**2 + (self.ball_y - self.sensor_pos_grid[j])**2)
                dist = torch.as_tensor(dist, dtype=torch.float32)
                sigma = 0.05  # m
                # Gaussian falloff
                f = torch.exp(-(dist**2) / (2.0 * sigma**2))
                # r: clamp normalized(ball_mass_n*f) between [0, max_firing_rate]
                r = torch.clamp((self.ball_mass_n * f) * self.max_firing_rate, 0.0, self.max_firing_rate)
                # per-step probability
                p[i][j] = torch.clamp(r * self.dt, 0.0, 1.0)
        pe = encoding.PoissonEncoder()
        inputs = pe(p)
        # if 1 in inputs:
            # print(inputs)
        if flatten:
            inputs = inputs.view(-1)  # (100,)
        return inputs
    
    def reset(self, seed=None, options=None):
        """
        INPUTS
            seed: 난수 시드 (int)
            options: (미사용)
        OUTPUTS
            observation: 초기 센서 입력 (10,10) 또는 (n_mf,) (옵션에 따라 다름)

        Reset ball mass, position, plate angles, and set xml
        """
        # print("Resetting environment...")
        # initialize all state variables
        self.step_count = 0
        self.success_count : int = 0
        if self.cfg.ball_mass is None:
            self.ball_mass = np.random.uniform(0.001, 0.03) #kg
            self.ball_mass_n = (self.ball_mass - 0.001) / 0.029  # [0,1]로 정규화
            self.ball_radius = self.ball_mass ** (1/3) / 10 #(m)
            
        self.ball_x = np.random.uniform(-self.plate_size*0.5, self.plate_size*0.5)
        self.ball_y = np.random.uniform(-self.plate_size*0.5, self.plate_size*0.5)

        print(f"ball mass={self.ball_mass*1000:.4f}g (norm = {self.ball_mass_n:.3f}), radius={self.ball_radius*100:.4f}cm ball pos=({self.ball_x*100:.4f}, {self.ball_y*100:.4f})cm")
     

        xml = f"""
        <mujoco model="tilt_plate">
        <compiler angle="degree" inertiafromgeom="true"/>
        <option timestep="{self.dt:.7f}" gravity="0 0 -9.81" integrator="RK4"/>
        <default>
            <geom  condim="6" margin="0.001" solimp="0.9 0.95 0.001" solref="0.002 1"/>
            <default class="plate">
            <geom type="box" friction="0.8 0.003 0.001" solimp="0.99 0.99 0.001" solref="0.01 1.5" rgba="0.8 0.8 0.85 1"/>
            </default>
            <default class="ball">
            <geom type="sphere" friction="0.9 0.005 0.002" rgba="0.9 0.3 0.3 1"/>
            </default>
            <joint armature="0.002" damping="0.1" limited="true"/>
            <motor gear="1.0" ctrllimited="true" ctrlrange="-1.0 1.0"/>
        </default>
        <worldbody>
            <body name="plate_base" pos="0 0 0">
            <joint name="hinge_x" type="hinge" axis="1 0 0" range="-5 5"/>
            <joint name="hinge_y" type="hinge" axis="0 1 0" range="-5 5"/>
            <geom name="plate_geom" class="plate" size="{self.plate_size/2} {self.plate_size/2} 0.005" mass="1.0"/>
            </body>
            <body name="ball" pos="{self.ball_x} {self.ball_y} {self.ball_radius+0.01:.4f}">
            <freejoint name="ball_free"/>
            <geom name="ball_geom" class="ball" size="{self.ball_radius}" mass="{self.ball_mass}"/>
            <!-- Marker for +x, -x -->
            <geom name="x_marker" type="box" size="0.02 0.005 0.003"
                pos="{self.plate_size/2} 0 0.01" rgba="1 0 0 1"
                group="3" contype="0" conaffinity="0"/>
            <geom name="y_marker" type="box" size="0.005 0.02 0.003"
                pos="0 {self.plate_size/2} 0.01" rgba="0 1 0 1"
                group="3" contype="0" conaffinity="0"/>
            </body>
        </worldbody>
        <actuator>
            <!-- ctrl가 '목표 각도[rad]'가 됨. kp↑, kv↑로 강한 서보 구성 -->
            <position name="px" joint="hinge_x" kp="500" kv="5" ctrlrange="-0.087 0.087"/>  <!-- ±5deg -->
            <position name="py" joint="hinge_y" kp="500" kv="5" ctrlrange="-0.087 0.087"/>
        </actuator>

        </mujoco>
        """.strip()
        self.model = mj.MjModel.from_xml_string(xml)
        self.data  = mj.MjData(self.model)
        obs = self.sensor_inputs(flatten=True).numpy().astype(np.float32)
        env_info = {
            "ball_pos": np.array([self.ball_x, self.ball_y], dtype=np.float32),
            "ball_mass": float(self.ball_mass_n)
        }
        return obs, env_info

    def step(self, action: np.ndarray):
        """
        INPUTS
            action: np.ndarray of shape (4,), values in [0,1]
                [XP, XN, YP, YN] - 각 모터에 대한 제어 신호
        
        OUTPUTS
            observation: 현재 시간 T에 대한 센서 입력 (n_mf,)
            reward: 보상 (float)
            3초 이상 가운데를 유지하면 terminated=True
            판을 벗어나면 truncated=True
            step_info: 환경 정보 반환 {
                ball_pos: 공 위치 (2,)
                ball_vel: 공 속도 (2,)
                plate_angles_rad: 판 경사각 (2,)
                step_count: 현재 스텝 수 (int)
            }

        """
        self.step_count += 1

        self.data.ctrl[0] = float(action[0] - action[1])  # X축 토크
        self.data.ctrl[1] = float(action[2] - action[3])  # Y축 토크
        mj.mj_step(self.model, self.data) # xml에서 정의한 dt만큼 단일 스텝 진행
        # ... (환경 단계 코드)
        # TODO: 관측, 보상, 종료, 정보 반환
        # self.ball_x, self.ball_y 업데이트
        ball_pos = self.data.body("ball").xpos  # (3,)
        self.ball_x, self.ball_y = ball_pos[0], ball_pos[1]

        # reward for distance - bonus if in success zone
        # dist = d-1cm
        dist_from_target = np.sqrt(self.ball_x**2 + self.ball_y**2) - 0.01
        reward = -dist_from_target * 100  # cm 단위로 보상
        # 
        if dist_from_target < 0:
            reward *= self.success_reward  # 보너스

        # hinge degreees - in radians
        hinge_x_q = self.data.joint("hinge_x").qpos.copy()
        hinge_y_q = self.data.joint("hinge_y").qpos.copy()

        observation = self.sensor_inputs(flatten=False).numpy()
        # terminated if stayed in center for success_timestep
        terminated = False
        if dist_from_target < 0:
            self.success_count += 1
            if self.success_count >= self.success_timestep:
                terminated = True
                reward += (self.simulation_duration_s/self.dt-self.step_count) * self.success_reward  # 보너스
                print(f"Success! Stayed in center for {self.success_timestep} steps.")

        # truncated if out of bounds
        if abs(self.ball_x)>0.15 or abs(self.ball_y)>0.15:  # 판 밖으로 나가면 종료
            reward -= self.failure_reward_penalty
            truncated = True
        else:
            truncated = False
        step_info = {
            # step information
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "plate_angles_rad": np.array([hinge_x_q, hinge_y_q], dtype=np.float32),
            # 메타
            "step_count": self.step_count
        }
        return observation, reward, terminated, truncated, step_info
    

  