1. Normalisation [−1,1] — matches N(0,1) diffusion noise; also tanh; also zero-centred training
2. Why hflip is the only augmentation — generative models reproduce their training distribution, so augmentation becomes output
3. Why the VAE is blurry — Gaussian likelihood → MSE → conditional mean → averaging. Plus: why MNIST hid it
4. The three distributional choices — Gaussian prior + Gaussian posterior → closed-form KL; Gaussian likelihood → MSE. And what each alternative would cost
5. β — not in the ELBO; β=1 is the true bound; Higgins' Lagrange-multiplier framing
6. log_var not std — positivity for free, numerical range, KL formula alignment
7. The reparameterization trick — can't backprop through sampling; move randomness to an input; it's a 1-sample MC estimator of the expectation
8. latent_dim — chosen, not derived; the compression ratio; inactive-units diagnostic
9. 64×64 — compute scales ~quadratically; pixel-space diffusion works cleanly at this size
