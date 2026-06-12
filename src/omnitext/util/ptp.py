import abc
import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import AttnProcessor2_0
from omnitext.technique.attend_and_excite import AttendExciteCrossAttnProcessor

def register_attention_control(unet, controller):
    attn_procs = {}
    attn_count = 0
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
            place_in_unet = "down"
        else:
            continue
        if 'attn2' in name:
            attn_count += 1
            attn_procs[name] = AttendExciteCrossAttnProcessor(
                attnstore=controller, place_in_unet=place_in_unet
            )
            print(name, place_in_unet)
        else:
            attn_procs[name] = AttnProcessor2_0()
            
    
    unet.set_attn_processor(attn_procs)
    controller.num_att_layers = attn_count

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            self.forward(attn, is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore:

    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [], "mid_cross": [], "up_cross": [],
            "down_self": [], "mid_self": [], "up_self": []
        }

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= 0 :
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            if is_cross:
                if attn.shape[1] in self.valid_res:
                    # print("HUHU SAVED CROSS", self.cur_att_layer, is_cross, place_in_unet)
                    self.step_store[key].append(attn)
                    self.cur_att_layer += 1 
            else:
                if attn.shape[2] in self.valid_res:
                    # print("HUHU SAVED SELF", self.cur_att_layer, is_cross, place_in_unet)
                    self.step_store[key].append(attn)
                    self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.curr_step_index += 1            
            self.between_steps()
            # print("HUHU RESET", self.curr_step_index, self.cur_att_layer, self.num_att_layers)


    def between_steps(self):
        self.attention_store = self.step_store
        del self.step_store
        torch.cuda.empty_cache()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def aggregate_attention(self, from_where: list[str], cutoff_index: int, is_cross: bool = True) -> torch.Tensor:
        out = []
        attention_maps = self.get_average_attention()
        if is_cross:
            for location in from_where:
                # print(len(attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]))
                for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                    spatial_size = item.shape[1]
                    if spatial_size == 256:
                        reshape_h = 16
                        reshape_w = 16
                    elif spatial_size == 1024:
                        reshape_h = 32
                        reshape_w = 32
                    target_h, target_w = 32, 32 # oom if it's 32 64
                    cross_maps = item[:, :, :] # :cutoff_index
                    cross_maps = cross_maps.reshape(-1, reshape_h, reshape_w, cross_maps.shape[-1]).permute(0, 3, 1, 2) # B H W D -> B D H W
                    cross_maps = cross_maps[:, :, :, :target_h] # crop to the region of interest
                    # print("YUHU B", cross_maps.size())
                    cross_maps = F.interpolate(cross_maps, (target_h, target_h)).permute(0, 2, 3, 1) # B D H W -> B H W D
                    # print("YUHU A", cross_maps.size())
                    out.append(cross_maps)
                    del cross_maps
            # print("HIYA", len(out))
            out = torch.cat(out, dim=0)
            # print(out.size())
            out = out.sum(0) / out.shape[0]
        else:
            for location in from_where:
                for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                    out.append(item)
                
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.curr_step_index = 0
        self.step_store = self.get_empty_store()
    
    def __init__(self, valid_res: list[int]):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.valid_res = valid_res
        


class GridAttentionStore:
    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [], "mid_cross": [], "up_cross": [],
            "down_self": [], "mid_self": [], "up_self": []
        }

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= 0 :
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            if is_cross:
                if attn.shape[1] in self.valid_res:
                    # print("HUHU SAVED CROSS", self.cur_att_layer, is_cross, place_in_unet)
                    self.step_store[key].append(attn)
                    self.cur_att_layer += 1 
            else:
                if attn.shape[2] in self.valid_res:
                    # print("HUHU SAVED SELF", self.cur_att_layer, is_cross, place_in_unet)
                    self.step_store[key].append(attn)
                    self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.curr_step_index += 1            
            self.between_steps()
            # print("HUHU RESET", self.curr_step_index, self.cur_att_layer, self.num_att_layers)


    def between_steps(self):
        self.attention_store = self.step_store
        del self.step_store
        torch.cuda.empty_cache()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def aggregate_attention(self, from_where: list[str], cutoff_index: int, is_cross: bool = True) -> torch.Tensor:
        out = []
        attention_maps = self.get_average_attention()
        if is_cross:
            for location in from_where:
                for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                    spatial_size = item.shape[1]
                    if spatial_size == 512:
                        reshape_h = 16
                        reshape_w = 32
                    elif spatial_size == 2048:
                        reshape_h = 32
                        reshape_w = 64
                    target_h, target_w = 32, 64 # oom if it's 32 64
                    cross_maps = item[:, :, :] # :cutoff_index
                    cross_maps = cross_maps.reshape(-1, reshape_h, reshape_w, cross_maps.shape[-1]).permute(0, 3, 1, 2) # B H W D -> B D H W
                    cross_maps = cross_maps[:, :, :, :target_h] # crop to the region of interest
                    # print("YUHU B", cross_maps.size())
                    cross_maps = F.interpolate(cross_maps, (target_h, target_h)).permute(0, 2, 3, 1) # B D H W -> B H W D
                    # print("YUHU A", cross_maps.size())
                    out.append(cross_maps)
                    del cross_maps
            # print("HIYA", len(out))
            out = torch.cat(out, dim=0)
            # print(out.size())
            out = out.sum(0) / out.shape[0]
        else:
            for location in from_where:
                for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                    out.append(item)
                
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.curr_step_index = 0
        self.step_store = self.get_empty_store()
    
    def __init__(self, valid_res: list[int]):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.valid_res = valid_res
        



    