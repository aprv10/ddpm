import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader


def show_images(dataset, num_samples=20, cols = 4):
  num_samples = min(num_samples, len(dataset))
  plt.figure(figsize=(15, 15))
  for i in range(num_samples):
    img, _ = dataset[i]
    plt.subplot(math.ceil(num_samples / cols), cols, i+1)
    plt.imshow(img, cmap="gray")
    plt.axis('off')
  plt.tight_layout()


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
  return torch.linspace(start, end, timesteps)


def get_ind_from_list(vals, t, x_shape):
  batch_size = t.shape[0]
  out = vals.to(t.device).gather(-1, t)
  return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def forward_diffusion_sample(x_0, t, device=None):
  if device is None:
    device = x_0.device
  x_0 = x_0.to(device)
  t = t.to(device)
  noise = torch.randn_like(x_0)
  sqrt_alphas_cumprod_t = get_ind_from_list(sqrt_alphas_cumprod, t, x_0.shape)
  sqrt_one_minus_alphas_cumprod_t = get_ind_from_list(
      sqrt_one_minus_alphas_cumprod, t, x_0.shape
  )

  return sqrt_alphas_cumprod_t * x_0 \
  + sqrt_one_minus_alphas_cumprod_t * noise, noise


T = 200
betas = linear_beta_schedule(timesteps=T)

alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

IMG_SIZE = 64
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001


def load_transformed_dataset():
  data_transforms = [
      transforms.Resize((IMG_SIZE, IMG_SIZE)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Lambda(lambda t: (t * 2) - 1)
  ]
  data_transform = transforms.Compose(data_transforms)

  train = torchvision.datasets.FashionMNIST(root=".", download=True, transform=data_transform, train=True)
  test = torchvision.datasets.FashionMNIST(root=".", download=True, transform=data_transform, train=False)

  return torch.utils.data.ConcatDataset([train, test])


def show_tensor_image(image):
  if len(image.shape) == 4:
    image = image[0, :, :, :]
  image = (image.detach().cpu().float().clamp(-1, 1) + 1) / 2
  if image.shape[0] == 1:
    plt.imshow(image[0], cmap="gray", vmin=0, vmax=1)
  else:
    plt.imshow(image.permute(1, 2, 0))


class Block(nn.Module):
  def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
    super().__init__()
    self.time_mlp = nn.Linear(time_emb_dim, out_ch)
    if up:
      self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
      self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
    else:
      self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
      self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    self.bnorm1 = nn.BatchNorm2d(out_ch)
    self.bnorm2 = nn.BatchNorm2d(out_ch)
    self.relu = nn.ReLU()

  def forward(self, x, t):
    h = self.bnorm1(self.relu(self.conv1(x)))
    time_emb = self.relu(self.time_mlp(t))
    h = h + time_emb[:, :, None, None]
    h = self.bnorm2(self.relu(self.conv2(h)))
    return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, time):
    half_dim = self.dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
    embeddings = time[:, None] * embeddings[None, :]
    return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class SimpleUnet(nn.Module):
  def __init__(self):
    super().__init__()
    image_channels = 1
    down_channels = (64, 128, 256, 512, 1024)
    up_channels = (1024, 512, 256, 128, 64)
    time_emb_dim = 32

    self.time_mlp = nn.Sequential(
        SinusoidalPositionEmbeddings(time_emb_dim),
        nn.Linear(time_emb_dim, time_emb_dim),
        nn.ReLU()
    )
    self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
    self.downs = nn.ModuleList([
        Block(down_channels[i], down_channels[i+1], time_emb_dim)
        for i in range(len(down_channels)-1)
    ])
    self.ups = nn.ModuleList([
        Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True)
        for i in range(len(up_channels)-1)
    ])
    self.output = nn.Conv2d(up_channels[-1], image_channels, 1)

  def forward(self, x, timestep):
    t = self.time_mlp(timestep)
    x = self.conv0(x)
    residual_inputs = []
    for down in self.downs:
      x = down(x, t)
      residual_inputs.append(x)
    for up in self.ups:
      residual_x = residual_inputs.pop()
      x = torch.cat((x, residual_x), dim=1)
      x = up(x, t)
    return self.output(x)


def get_loss(model, x_0, t):
  x_noisy, noise = forward_diffusion_sample(x_0, t)
  noise_pred = model(x_noisy, t)
  return F.l1_loss(noise_pred, noise)


@torch.no_grad()
def sample_timestep(model, x, t):
  betas_t = get_ind_from_list(betas, t, x.shape)
  sqrt_one_minus_alphas_cumprod_t = get_ind_from_list(
      sqrt_one_minus_alphas_cumprod, t, x.shape
  )
  sqrt_recip_alphas_t = get_ind_from_list(sqrt_recip_alphas, t, x.shape)
  with torch.autocast(device_type=x.device.type, enabled=x.is_cuda):
    noise_pred = model(x, t)
  model_mean = sqrt_recip_alphas_t * (
      x - betas_t * noise_pred.float() / sqrt_one_minus_alphas_cumprod_t
  )
  posterior_variance_t = get_ind_from_list(posterior_variance, t, x.shape)
  nonzero_mask = (t != 0).reshape(-1, 1, 1, 1)
  return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * torch.randn_like(x)


@torch.no_grad()
def sample_plot_image(model, num_samples=16):
  device = next(model.parameters()).device
  was_training = model.training
  model.eval()
  img = torch.randn((num_samples, 1, IMG_SIZE, IMG_SIZE), device=device)
  snapshots = [img[:1].cpu()]
  snapshot_steps = set(torch.linspace(T-1, 0, 9).long().tolist())

  for i in reversed(range(T)):
    t = torch.full((num_samples,), i, device=device, dtype=torch.long)
    img = sample_timestep(model, img, t)
    if i in snapshot_steps:
      snapshots.append(img[:1].cpu())

  plt.figure(figsize=(15, 3))
  for i, image in enumerate(snapshots):
    plt.subplot(1, len(snapshots), i+1)
    show_tensor_image(image)
    plt.axis('off')
  plt.tight_layout()
  plt.savefig("reverse_diffusion.png")
  plt.close()

  cols = math.ceil(math.sqrt(num_samples))
  plt.figure(figsize=(8, 8))
  for i, image in enumerate(img):
    plt.subplot(math.ceil(num_samples / cols), cols, i+1)
    show_tensor_image(image)
    plt.axis('off')
  plt.tight_layout()
  plt.savefig("generated_images.png")
  plt.close()
  model.train(was_training)
  return img.detach().cpu()


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using {device}")

  data = torchvision.datasets.FashionMNIST(root=".", download=True)
  show_images(data)
  plt.savefig("fashion_mnist.png")
  plt.close()

  data = load_transformed_dataset()
  dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=True, pin_memory=device.type == "cuda")

  image = next(iter(dataloader))[0][:1]
  num_images = 10
  plt.figure(figsize=(15, 3))
  plt.subplot(1, num_images+1, 1)
  show_tensor_image(image)
  plt.title("Original")
  plt.axis('off')
  for i, idx in enumerate(torch.linspace(0, T-1, num_images).long().tolist()):
    t = torch.tensor([idx], dtype=torch.long)
    noisy_image, _ = forward_diffusion_sample(image, t)
    plt.subplot(1, num_images+1, i+2)
    show_tensor_image(noisy_image)
    plt.title(f"t={idx}")
    plt.axis('off')
  plt.tight_layout()
  plt.savefig("forward_diffusion.png")
  plt.close()

  model = SimpleUnet().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, foreach=False)
  scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
  losses = []

  for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.
    for batch, _ in dataloader:
      optimizer.zero_grad(set_to_none=True)
      x_0 = batch.to(device, non_blocking=True)
      t = torch.randint(0, T, (x_0.shape[0],), device=device, dtype=torch.long)
      with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        loss = get_loss(model, x_0, t)
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      epoch_loss += loss.item()

    losses.append(epoch_loss / len(dataloader))
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {losses[-1]:.4f}")
    torch.save(model.state_dict(), "ddpm_model.pth")

  optimizer.zero_grad(set_to_none=True)
  plt.figure()
  plt.plot(range(1, EPOCHS+1), losses)
  plt.xlabel("Epoch")
  plt.ylabel("L1 noise prediction loss")
  plt.tight_layout()
  plt.savefig("training_loss.png")
  plt.close()
  sample_plot_image(model)


if __name__ == "__main__":
  main()
