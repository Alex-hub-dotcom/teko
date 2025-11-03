class PolicyNetwork(GaussianMixin, Model):
    """Policy: RGB -> wheel velocities"""
    
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, **kwargs)
        
        # Usa o teu CNN encoder
        self.encoder = create_visual_encoder(
            architecture="simple",  # Começa com SimpleCNN (mais rápido)
            feature_dim=256,
            pretrained=False
        )
        
        # Cabeça da policy
        self.policy = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
            nn.Tanh()  # Ações em [-1, 1]
        )
        
        # Log std para a política Gaussiana
        self.log_std = nn.Parameter(torch.zeros(self.num_actions))
        
    def compute(self, inputs, role):
        # SKRL passa observações como dicionário
        states = inputs["states"]
        
        # Extrai RGB
        if isinstance(states, dict):
            if "policy" in states:
                rgb = states["policy"]["rgb"]
            elif "rgb" in states:
                rgb = states["rgb"]
            else:
                raise ValueError(f"Cannot find RGB in states: {states.keys()}")
        else:
            rgb = states
        
        # Ajusta o formato dependendo da entrada
        if rgb.dim() == 2:
            # Flattened [batch*pixels, channels] -> [batch, channels, H, W]
            total_pixels = 480 * 640
            batch_size = rgb.shape[0] // total_pixels
            rgb = rgb.view(batch_size, total_pixels, 3)  # [B, pixels, C]
            rgb = rgb.permute(0, 2, 1)                   # [B, C, pixels]
            rgb = rgb.view(batch_size, 3, 480, 640)      # [B, C, H, W]
        elif rgb.dim() == 3:
            # [batch, pixels, channels] -> [batch, channels, H, W]
            batch_size = rgb.shape[0]
            rgb = rgb.permute(0, 2, 1)
            rgb = rgb.view(batch_size, 3, 480, 640)
        # else: já está no formato [B, 3, 480, 640]
        
        # Extrai features visuais
        features = self.encoder(rgb)
        
        # Passa pelas camadas fully connected
        actions = self.policy(features)
        
        return actions, self.log_std, {}
