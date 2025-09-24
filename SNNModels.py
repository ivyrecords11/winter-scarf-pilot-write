from mujoco_model import Environment, SimulationConfig
import torch
import torch.nn as nn
import SpikingJelly
from SpikingJelly.activation_based import layer, neuron, functional, learning, surrogate, encoding

class CerebellarNet:
    def __init__(self, cfg: SimulationConfig):
        self.cfg        = cfg
        self.dt         = cfg.dt  # 초 단위로 변환
        self.n_mf       = cfg.n_mf
        self.n_grc      = cfg.n_grc
        self.n_pkj      = cfg.n_pkj
        self.n_cf       = cfg.n_cf
        self.n_motor = cfg.n_motor
        # 시냅스 시간효율
        self.tau_e_ms = cfg.tau_e_ms
        self.tau_e_ms_cf = cfg.tau_e_ms
        # ... (SNN 초기화 코드)
        self.mf2grc     = layer.Linear(self.n_mf, self.n_grc, bias=False)
        self.grc        = neuron.LIFNode(tau=self.dt*self.tau_e_ms, surrogate_function=surrogate.ATan())
        self.grc2pkj    = layer.Linear(self.n_grc, self.n_pkj, bias=False)
        self.pkj        = neuron.LIFNode(tau=self.dt*self.tau_e_ms, surrogate_function=surrogate.ATan())
        self.pkj2mo     = layer.Linear(self.n_pkj, self.n_motor, bias=False)
        self.mo         = neuron.LIFNode(tau=self.dt*self.tau_e_ms, surrogate_function=surrogate.ATan())
        '''
        self.cf2pkj     = layer.Linear(self.n_cf, self.n_pkj, bias=False)
        self.cf         = neuron.LIFNode(tau=self.dt*self.tau_e_ms, surrogate_function=surrogate.ATan())'''
        
        self.net = nn.sequential(
            self.mf2grc,
            self.grc,
            self.grc2pkj,
            self.pkj,
            self.pkj2mo,
            self.mo
        )
        functional.set_step_mode(self.net, 'm')
    def forward(self, mf_spikes):
        """
        x_bool_bchw: shape (B, 1, 10, 10), dtype float32 in {0.,1.}
        return: logits shape (B, 4)
        """
        # ... (SNN 순전파 코드)
        mf_spikes = mf_spikes.reshape(-1, self.n_mf)  # (B, n_mf)
        return self.net(mf_spikes)
    @torch.no_grad()
    def act_inference(self, mf_spikes):
        return self.net(mf_spikes)


class CerebellarCNN:
    def __init__(self, cfg: SimulationConfig):
        self.cfg        = cfg
        self.dt         = cfg.dt  # 초 단위로 변환
        self.cnn1channels = 16
        self.cnn1width = 5
        self.cnn2channels = 32
        self.fc_channels = 128
        self.n_motor = cfg.n_motor
        # 시냅스 시간효율
        self.tau_e_ms = cfg.tau_e_ms
        self.tau_e_ms_cf = cfg.tau_e_ms
        # ... (SNN 초기화 코드)
    def forward(self, mf_spikes):
        # ... (SNN 순전파 코드)
        motor_commands = []
        return motor_commands

