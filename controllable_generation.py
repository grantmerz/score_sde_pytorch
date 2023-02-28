from models import utils as mutils
import torch
import torch.nn as nn
import numpy as np
from sampling import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
import functools
import time


def get_pc_inpainter(sde, predictor, corrector, inverse_scaler, snr,
                     n_steps=1, probability_flow=False, continuous=False,
                     denoise=True, eps=1e-5):
  """Create an image inpainting function that uses PC samplers.


  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for the corrector.
    n_steps: An integer. The number of corrector steps per update of the corrector.
    probability_flow: If `True`, predictor solves the probability flow ODE for sampling.
    continuous: `True` indicates that the score-based model was trained with continuous time.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The reverse-time SDE/ODE is integrated to `eps` for numerical stability.

  Returns:
    An inpainting function.
  """
  # Define predictor & corrector
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def get_inpaint_update_fn(update_fn):
    """Modify the update function of predictor & corrector to incorporate data information."""

    def inpaint_update_fn(model, data, mask, x, t):
      with torch.no_grad():
        vec_t = torch.ones(data.shape[0], device=data.device) * t
        x, x_mean = update_fn(x, vec_t, model=model)
        masked_data_mean, std = sde.marginal_prob(data, vec_t)
        masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
        x = x * (1. - mask) + masked_data * mask
        x_mean = x * (1. - mask) + masked_data_mean * mask
        return x, x_mean

    return inpaint_update_fn

  projector_inpaint_update_fn = get_inpaint_update_fn(predictor_update_fn)
  corrector_inpaint_update_fn = get_inpaint_update_fn(corrector_update_fn)

  def pc_inpainter(model, data, mask):
    """Predictor-Corrector (PC) sampler for image inpainting.

    Args:
      model: A score model.
      data: A PyTorch tensor that represents a mini-batch of images to inpaint.
      mask: A 0-1 tensor with the same shape of `data`. Value `1` marks known pixels,
        and value `0` marks pixels that require inpainting.

    Returns:
      Inpainted (complete) images.
    """
    with torch.no_grad():
      # Initial sample
      x = data * mask + sde.prior_sampling(data.shape).to(data.device) * (1. - mask)
      timesteps = torch.linspace(sde.T, eps, sde.N)
      for i in range(sde.N):
        t = timesteps[i]
        x, x_mean = corrector_inpaint_update_fn(model, data, mask, x, t)
        x, x_mean = projector_inpaint_update_fn(model, data, mask, x, t)

      return inverse_scaler(x_mean if denoise else x)

  return pc_inpainter


def get_pc_colorizer(sde, predictor, corrector, inverse_scaler,
                     snr, n_steps=1, probability_flow=False, continuous=False,
                     denoise=True, eps=1e-5):
  """Create a image colorization function based on Predictor-Corrector (PC) sampling.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    predictor: A subclass of `sampling.Predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.Corrector` that represents a corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for correctors.
    n_steps: An integer. The number of corrector steps per update of the predictor.
    probability_flow: If `True`, solve the probability flow ODE for sampling with the predictor.
    continuous: `True` indicates that the score-based model was trained with continuous time steps.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The SDE/ODE will start from `eps` to avoid numerical stabilities.

  Returns: A colorization function.
  """

  # `M` is an orthonormal matrix to decouple image space to a latent space where the gray-scale image
  # occupies a separate channel
  M = torch.tensor([[5.7735014e-01, -8.1649649e-01, 4.7008697e-08],
                   [5.7735026e-01, 4.0824834e-01, 7.0710671e-01],
                   [5.7735026e-01, 4.0824822e-01, -7.0710683e-01]])
  # `invM` is the inverse transformation of `M`
  invM = torch.inverse(M)

  # Decouple a gray-scale image with `M`
  def decouple(inputs):
    return torch.einsum('bihw,ij->bjhw', inputs, M.to(inputs.device))

  # The inverse function to `decouple`.
  def couple(inputs):
    return torch.einsum('bihw,ij->bjhw', inputs, invM.to(inputs.device))

  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def get_colorization_update_fn(update_fn):
    """Modify update functions of predictor & corrector to incorporate information of gray-scale images."""

    def colorization_update_fn(model, gray_scale_img, x, t):
      mask = get_mask(x)
      vec_t = torch.ones(x.shape[0], device=x.device) * t
      x, x_mean = update_fn(x, vec_t, model=model)
      masked_data_mean, std = sde.marginal_prob(decouple(gray_scale_img), vec_t)
      masked_data = masked_data_mean + torch.randn_like(x) * std[:, None, None, None]
      x = couple(decouple(x) * (1. - mask) + masked_data * mask)
      x_mean = couple(decouple(x) * (1. - mask) + masked_data_mean * mask)
      return x, x_mean

    return colorization_update_fn

  def get_mask(image):
    mask = torch.cat([torch.ones_like(image[:, :1, ...]),
                      torch.zeros_like(image[:, 1:, ...])], dim=1)
    return mask

  predictor_colorize_update_fn = get_colorization_update_fn(predictor_update_fn)
  corrector_colorize_update_fn = get_colorization_update_fn(corrector_update_fn)

  def pc_colorizer(model, gray_scale_img):
    """Colorize gray-scale images using Predictor-Corrector (PC) sampler.

    Args:
      model: A score model.
      gray_scale_img: A minibatch of gray-scale images. Their R,G,B channels have same values.

    Returns:
      Colorized images.
    """
    with torch.no_grad():
      shape = gray_scale_img.shape
      mask = get_mask(gray_scale_img)
      # Initial sample
      x = couple(decouple(gray_scale_img) * mask + \
                 decouple(sde.prior_sampling(shape).to(gray_scale_img.device)
                          * (1. - mask)))
      timesteps = torch.linspace(sde.T, eps, sde.N)
      for i in range(sde.N):
        t = timesteps[i]
        x, x_mean = corrector_colorize_update_fn(model, gray_scale_img, x, t)
        x, x_mean = predictor_colorize_update_fn(model, gray_scale_img, x, t)

      return inverse_scaler(x_mean if denoise else x)

  return pc_colorizer



def get_mixture_conditional_sampler(sde,
                               predictor, corrector, inverse_scaler, snr,
                               n_steps=1, probability_flow=False,
                               continuous=False, denoise=True, eps=1e-5,M=1):
  """Class-conditional sampling with Predictor-Corrector (PC) samplers.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    score_model: A `flax.linen.Module` object that represents the architecture of the score-based model.
    classifier: A `flax.linen.Module` object that represents the architecture of the noise-dependent classifier.
    classifier_params: A dictionary that contains the weights of the classifier.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.predictor` that represents a predictor algorithm.
    corrector: A subclass of `sampling.corrector` that represents a corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for correctors.
    n_steps: An integer. The number of corrector steps per update of the predictor.
    probability_flow: If `True`, solve the probability flow ODE for sampling with the predictor.
    continuous: `True` indicates the score-based model was trained with continuous time.
    denoise: If `True`, add one-step denoising to final samples.
    eps: A `float` number. The SDE/ODE will be integrated to `eps` to avoid numerical issues.
  Returns: A pmapped class-conditional image sampler.
  """
  # A function that gives the logits of the noise-dependent classifier
  #logit_fn = mutils.get_logit_fn(classifier, classifier_params)
  # The gradient function of the noise-dependent classifier
  #@torch.enable_grad()
  def mixture_grad_fn(x1t, x2t, ve_noise_scale, mixed):
    lambda_ = 1./(ve_noise_scale**2)
    lam = lambda_[0]
    with torch.enable_grad():

      x1n = x1t.clone().detach().requires_grad_(True)
      x2n = x2t.clone().detach().requires_grad_(True)
        
      xs=[x1n,x2n]
      recon_loss = (torch.norm(torch.flatten(sum(xs) - mixed)) ** 2)
      recon_grads = torch.autograd.grad(recon_loss, xs)
      mult = torch.einsum ('i, ijkl -> ijkl', lambda_, recon_grads[0])
    
    #mult = lam*(x1t + x2t - mixed)

    return mult



  #classifier_grad_fn = mutils.get_classifier_grad_fn(logit_fn)
  #mixture_grad_fn = get_mixture_grad_fn(x1, x2, ve_noise_scale, mixed)

  def conditional_predictor_update_fn(model, x1, x2, t, mixed):
    """The predictor update function for class-conditional sampling."""
    score_fn = mutils.get_score_fn(sde, model, train=False,
                                   continuous=continuous)

    

    def total_grad_fn(x1, t):
      ve_noise_scale = sde.marginal_prob(x1, t)[1]
      #return score_fn(x, t) + classifier_grad_fn(x, ve_noise_scale, labels)
      return score_fn(x1, t) - mixture_grad_fn(x1,x2, ve_noise_scale, mixed)


    if predictor is None:
      predictor_obj = NonePredictor(sde, total_grad_fn, probability_flow)
    else:
      predictor_obj = predictor(sde, total_grad_fn, probability_flow)
    return predictor_obj.update_fn(x1, t)

  def conditional_corrector_update_fn(model, x1t, x2t, t, mixed):
    """The corrector update function for class-conditional sampling."""
    score_fn = mutils.get_score_fn(sde, model, train=False,
                                   continuous=continuous)

    def total_grad_fn(x1t, t):
      ve_noise_scale = sde.marginal_prob(x1t, t)[1]
      return score_fn(x1t, t) - mixture_grad_fn(x1t, x2t, ve_noise_scale, mixed)

    if corrector is None:
      corrector_obj = NoneCorrector(sde, total_grad_fn, snr, n_steps)
    else:
      corrector_obj = corrector(sde, total_grad_fn, snr, n_steps)
    return corrector_obj.update_fn(x1t, t)

  def pc_mixture_sampler(model, mixture):
    """Generate class-conditional samples with Predictor-Corrector (PC) samplers.
    Args:
      rng: A JAX random state.
      score_state: A `flax.struct.dataclass` object that represents the training state
        of the score-based model.
      labels: A JAX array of integers that represent the target label of each sample.
    Returns:
      Class-conditional samples.
    """
    with torch.no_grad():
      shape = mixture.shape
      #mask = get_mask(gray_scale_img)
      # Initial sample
      x1 = sde.prior_sampling(shape).to(mixture.device)
      x2 = sde.prior_sampling(shape).to(mixture.device)

      #x1=nn.Parameter(torch.Tensor(shape).uniform_()).to(mixture.device)
      #x2=nn.Parameter(torch.Tensor(shape).uniform_()).to(mixture.device)

      timesteps = torch.linspace(sde.T, eps, sde.N)
        
      #Try sampling at only a fraction of the noise scales
      #so that nsteps can be increased  
       
      for i,ts in enumerate(timesteps[::111]):
      #for i in range(sde.N):
        #t = timesteps[i]
        #vec_t = torch.ones(x1.shape[0], device=x1.device) * t
        vec_t = torch.ones(x1.shape[0], device=x1.device) * ts

        #This does all updates for x1 at one noise scale, then all for x2.  Don't want that!
        #x1c, x1_mean = conditional_corrector_update_fn(model, x1, x2, vec_t, mixture)
        #x2c, x2_mean = conditional_corrector_update_fn(model, x2, x1, vec_t, mixture)
        
        #Keep n_steps=1 and do multiple steps in another loop
        for i in range(M):
          x1_0 = x1
          x2_0 = x2
          x1, x1_mean = conditional_corrector_update_fn(model, x1_0, x2_0, vec_t, mixture)
          x2, x2_mean = conditional_corrector_update_fn(model, x2_0, x1_0, vec_t, mixture)
          
          #x1 = torch.clamp(x1, 0, 1)
          #x2 = torch.clamp(x2, 0, 1)
          #x1_mean = torch.clamp(x1_mean, 0, 1)
          #x2_mean = torch.clamp(x2_mean, 0, 1)

          #x1, x1_mean = conditional_predictor_update_fn(model, x1c, x2c, vec_t, mixture)
          #x2, x2_mean = conditional_predictor_update_fn(model, x2c, x1c, vec_t, mixture)

          #x1 = torch.clamp(x1, 0, 1)
          #x2 = torch.clamp(x2, 0, 1)
          #x1_mean = torch.clamp(x1_mean, 0, 1)
          #x2_mean = torch.clamp(x2_mean, 0, 1)

        #x1c = torch.clamp(x1c, 0, 1)
        #x2c = torch.clamp(x2c, 0, 1)


        x1c = x1
        x2c = x2
        x1, x1_mean = conditional_predictor_update_fn(model, x1c, x2c, vec_t, mixture)
        x2, x2_mean = conditional_predictor_update_fn(model, x2c, x1c, vec_t, mixture)

        #x1 = torch.clamp(x1, 0, 1)
        #x1_mean = torch.clamp(x1_mean,0,1)
        #x2 = torch.clamp(x2, 0, 1)
        #x2_mean = torch.clamp(x2_mean,0,1)


      return inverse_scaler(x1_mean if denoise else x1), inverse_scaler(x2_mean if denoise else x2)

  return pc_mixture_sampler



  '''
  def pc_conditional_sampler(rng, score_state, labels):
    """Generate class-conditional samples with Predictor-Corrector (PC) samplers.
    Args:
      rng: A JAX random state.
      score_state: A `flax.struct.dataclass` object that represents the training state
        of the score-based model.
      labels: A JAX array of integers that represent the target label of each sample.
    Returns:
      Class-conditional samples.
    """
    # Initial sample
    rng, step_rng = random.split(rng)
    x = sde.prior_sampling(step_rng, shape)

    timesteps = jnp.linspace(sde.T, eps, sde.N)

    def loop_body(i, val):
      rng, x, x_mean = val
      t = timesteps[i]
      vec_t = jnp.ones(shape[0]) * t
      rng, step_rng = random.split(rng)
      x, x_mean = conditional_corrector_update_fn(step_rng, score_state, x, vec_t, labels)
      rng, step_rng = random.split(rng)
      x, x_mean = conditional_predictor_update_fn(step_rng, score_state, x, vec_t, labels)
      return rng, x, x_mean

    _, x, x_mean = jax.lax.fori_loop(0, sde.N, loop_body, (rng, x, x))
    return inverse_scaler(x_mean if denoise else x)

  return jax.pmap(pc_conditional_sampler, axis_name='batch')


  '''