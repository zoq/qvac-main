# Important: img2img Dimensions

## The Issue

When using `model.img2img()`, **DO NOT specify `width` or `height` parameters**. 

If you do, you'll get this error:
```
GGML_ASSERT(image.width == tensor->ne[0]) failed
```

## Why?

For img2img, stable-diffusion.cpp automatically detects the dimensions from the input image. When you manually specify width/height, it creates a mismatch between:
- The latent tensor size (based on width/height params)
- The actual input image dimensions

## Correct Usage

```javascript
// ✅ CORRECT - No width/height
await model.img2img({
  prompt: 'professional headshot',
  init_image: imageBuffer,
  strength: 0.5,
  steps: 20
})

// ❌ WRONG - Specifying width/height causes crash
await model.img2img({
  prompt: 'professional headshot',
  init_image: imageBuffer,
  strength: 0.5,
  steps: 20,
  width: 800,   // DON'T DO THIS
  height: 800   // DON'T DO THIS
})
```

## Resolution Requirements

Your input image should already be:
- A multiple of 8 in both dimensions (e.g., 512, 640, 768, 800, 1024)
- Within FLUX2's supported range (up to 1024x1024 works well)

If your image isn't a multiple of 8, stable-diffusion.cpp will handle it internally.

## CLI vs JavaScript API

Note that the **CLI** (`sd-cli`) works differently:
- CLI: You CAN specify `--width` and `--height` (they resize the init image)
- JavaScript API: You CANNOT specify width/height (auto-detected only)

This is why `scripts/headshot.sh` works with width/height but the JavaScript example doesn't.
