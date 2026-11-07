import torch
from ddpm_scratch import SimpleUnet, sample_plot_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleUnet().to(device)
model.load_state_dict(
    torch.load("ddpm_model.pth", map_location=device, weights_only=True)
)
model.eval()

images = sample_plot_image(model, num_samples=16)