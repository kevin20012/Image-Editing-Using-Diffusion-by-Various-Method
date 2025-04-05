import torch

alpha = 0.1
is_complexity = False

def register_attention_control(model, controller=None):
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None,scale=1.0):
            is_cross = encoder_hidden_states is not None    
            #residual save            
            residual = hidden_states
            query = self.to_q(hidden_states) * 1.0

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)
            
                

            key = self.to_k(encoder_hidden_states) * 1.0
            value = self.to_v(encoder_hidden_states) * 1.0

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            baddbmm_input = None
            attention_probs = self.get_attention_scores(query, key, attention_mask=baddbmm_input)
            
            controller(attention_probs, is_cross, place_in_unet)

            hidden_states = torch.matmul(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)                
            hidden_states = self.to_out[0](hidden_states) * 1.0
            hidden_states = self.to_out[1](hidden_states)



            ours = True
            if ours:
                mid_scale, down_scale = alpha, alpha
            
                if self.to_k.in_features != self.to_q.in_features:
                    if place_in_unet == "down": 
                        if self.to_q.in_features == 640:
                            hidden_states = (1-down_scale)*hidden_states + residual*(down_scale)
                            pass
                        else:
                            hidden_states = (1-down_scale)*hidden_states + residual*(down_scale)

                            pass
                    #-----------------------------------------------------------------------------------------------
                    elif place_in_unet == "mid": 
                            hidden_states = (1- mid_scale)*hidden_states + residual*(mid_scale)
                            

                            pass
                    #-----------------------------------------------------------------------------------------------
                    elif place_in_unet == "up":
                        if self.to_q.in_features == 640:
                            pass
                        else:
                            pass
                #------------------------------------cross attention------------------------------------------------------
                else:
                #-------------------------------------self attention ----------------------------------------------------        
                    if place_in_unet == "down": 
                        if self.to_q.in_features == 640:
                            hidden_states = (1-down_scale)*hidden_states + residual*(down_scale)
                            pass
                        else:
                            hidden_states = (1-down_scale)*hidden_states + residual*(down_scale)
                            pass
                    #-----------------------------------------------------------------------------------------------
                    elif place_in_unet == "mid": 
                            hidden_states = (1- mid_scale)*hidden_states + residual*(mid_scale)
                            pass
                    #-----------------------------------------------------------------------------------------------
                    elif place_in_unet == "up":
                        if self.to_q.in_features == 640:
                            if is_complexity:
                                hidden_states = (1-down_scale)*hidden_states + residual*(down_scale)
                            pass
                        else:
                            pass
                    
            
            return hidden_states 

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        print("Dummy Controller Declaration because there is no Controller")
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count