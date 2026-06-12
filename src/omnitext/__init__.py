"""OmniText: A Unified Training-Free Framework for Controllable Text Image Manipulation.

Importable library backing the application scripts in ``scripts/``. Submodules:

* ``omnitext.model``      - model loading (CLIP / VAE / TextDiffuser-2 UNet)
* ``omnitext.pipeline``   - diffusion pipelines (style transfer + removal variant)
* ``omnitext.util``       - attention store (ptp), text and misc utilities
* ``omnitext.technique``  - attend-and-excite building blocks
* ``omnitext.char_mask``  - character-width tables and mask generation
* ``omnitext.frequency``  - frequency-domain low-pass filters
* ``omnitext.optim``      - PCGrad multi-task gradient surgery
* ``omnitext.data``       - OmniText-Bench loading helpers

Heavy dependencies (torch, diffusers) are imported lazily by the submodules, so
``import omnitext`` itself stays cheap.
"""

__version__ = "1.0.0"
