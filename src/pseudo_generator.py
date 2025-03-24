"""
Initial pseudo-mask generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logging.basicConfig(
    level=logging.INFO,               
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from einops import rearrange


class PseudoGenerator(nn.Module):
    def __init__(self, feature_layer, embedding_layer, img_size, backbone):
        super(PseudoGenerator, self).__init__()
        logging.info('build initial pseudo-mask generator')
        self.feature_layer = feature_layer
        self.embedding_layer = embedding_layer
        
        self.backbone = backbone.image_encoder
        self.backbone.eval()
        
        self.img_size = img_size
        self.patch_size = 16 
        
        
    def __forward(self, img, return_qkv=False):
        ## intercept feature facet
        qkv = self.backbone.get_intermediate_layers(
            img, n=32, return_qkv=return_qkv, lyr=self.feature_layer
        )
        return qkv


    def forward(self, inputs):
        input_t0, input_t1 = torch.split(inputs, 3, 1)
        with torch.no_grad():
            input_t0_qkv = self.__forward(input_t0, return_qkv=True)
            input_t1_qkv = self.__forward(input_t1, return_qkv=True)

            key, query, value = self._generate(input_t0_qkv, input_t1_qkv)
            
            _, embeds_t0 = self.backbone(input_t0)
            _, embeds_t1 = self.backbone(input_t1)
            
            embed_t0 = embeds_t0[self.embedding_layer-1].permute(0,3,1,2)
            embed_t1 = embeds_t1[self.embedding_layer-1].permute(0,3,1,2)
            
            embed_t0 = F.interpolate(embed_t0, self.img_size, mode='bilinear', align_corners=True).squeeze(0).permute(1,2,0)
            embed_t1 = F.interpolate(embed_t1, self.img_size, mode='bilinear', align_corners=True).squeeze(0).permute(1,2,0)
            
        return embed_t0, embed_t1, key, query, value
    
        
    def _generate(self, input_t0_qkv, input_t1_qkv):
        ## multi-head feature correlation
        _, B, N, L, C = input_t0_qkv.shape
        
        # seperate qkv
        input_t0_key = input_t0_qkv[1, :, :, :, :]
        input_t1_key = input_t1_qkv[1, :, :, :, :]
        input_t0_qry = input_t0_qkv[0, :, :, :, :]
        input_t1_qry = input_t1_qkv[0, :, :, :, :]
        input_t0_val = input_t0_qkv[2, :, :, :, :]
        input_t1_val = input_t1_qkv[2, :, :, :, :]
        
        h = int(self.img_size[0] // self.patch_size)
        w = int(self.img_size[1] // self.patch_size)
        
        # reshape to 2D image space
        input_t0_key = rearrange(input_t0_key, 'b n (h w) c -> b n h w c', h=h, w=w)
        input_t1_key = rearrange(input_t1_key, 'b n (h w) c -> b n h w c', h=h, w=w)
        input_t0_qry = rearrange(input_t0_qry, 'b n (h w) c -> b n h w c', h=h, w=w)
        input_t1_qry = rearrange(input_t1_qry, 'b n (h w) c -> b n h w c', h=h, w=w)
        input_t0_val = rearrange(input_t0_val, 'b n (h w) c -> b n h w c', h=h, w=w)
        input_t1_val = rearrange(input_t1_val, 'b n (h w) c -> b n h w c', h=h, w=w)
        
        # l2 normalization
        input_t0_key = F.normalize(input_t0_key, p=2, dim=-1)
        input_t1_key = F.normalize(input_t1_key, p=2, dim=-1)
        input_t0_qry = F.normalize(input_t0_qry, p=2, dim=-1)
        input_t1_qry = F.normalize(input_t1_qry, p=2, dim=-1)
        input_t0_val = F.normalize(input_t0_val, p=2, dim=-1)
        input_t1_val = F.normalize(input_t1_val, p=2, dim=-1)
        
        # calculate cosine similiarity
        key = torch.einsum('b n h w c, b n h w c -> b n h w', input_t0_key, input_t1_key)    
        query = torch.einsum('b n h w c, b n h w c -> b n h w', input_t0_qry, input_t1_qry)
        value = torch.einsum('b n h w c, b n h w c -> b n h w', input_t0_val, input_t1_val)
        
        # average in head dimension
        key = key.mean(dim=1, keepdim=True)
        query = query.mean(dim=1, keepdim=True)
        value = value.mean(dim=1, keepdim=True)
        
        # interpolate to original image size
        key = F.interpolate(key, self.img_size, mode='bilinear', align_corners=True).squeeze(1)
        query = F.interpolate(query, self.img_size, mode='bilinear', align_corners=True).squeeze(1)
        value = F.interpolate(value, self.img_size, mode='bilinear', align_corners=True).squeeze(1)
        
        return key, query, value
    