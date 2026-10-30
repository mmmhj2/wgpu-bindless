
# Bindless Renderer

A real-time renderer written in Rust using WebGPU (wgpu) with WGSL shaders, featuring bindless resource management, PBR materials with diffuse/normal/MRAO maps, and a modern rasterization pipeline.

## TODO LIST:

Short term:
- [ ] Rework asset GPU upload path;
- [ ] Improve skeletal animation quality (frame rates, runtime switching, etc.);
- [ ] Better bindless resource management (generation based handle, etc.)
- [ ] Rework frame structure and related borrowings.

Long term:
- [ ] Introduce a proper render graph (may not be worth it, since current passes are very simple);
- [ ] Fancier ligthing and shadow rendering;
- [ ] Dynamic vertex format matching with Naga.
