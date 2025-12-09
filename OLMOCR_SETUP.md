# oLmOCR Setup Guide for Jetson Orin

## Overview

oLmOCR uses **Qwen2.5-VL**, a modern LLM-based vision-language model that excels at:
- ✅ **Vertical text recognition** (excellent for your use case)
- ✅ Complex layouts and multi-column documents
- ✅ Handwritten text
- ✅ Multi-language support

**Important Notes:**
- oLmOCR is more resource-intensive than EasyOCR/PaddleOCR
- Recommended for edge devices: Use **Qwen2.5-VL-3B-Instruct** (smallest in Qwen2.5-VL series)
- Alternative: **Qwen2-VL-2B** (older version, smaller model ~2GB)
- For better accuracy: Use **Qwen2.5-VL-7B-Instruct** (requires more GPU memory)

## Installation

### 1. Install Dependencies

```bash
# On Jetson Orin
# Install transformers (latest version recommended for Qwen2.5-VL support)
pip3 install git+https://github.com/huggingface/transformers
# Or use stable version (may work but less features):
# pip3 install transformers>=4.49.0

pip3 install torch torchvision
pip3 install qwen-vl-utils
```

**Important:** For Qwen2.5-VL models, it's recommended to install transformers from source to get the latest features and bug fixes.

**Note:** First run will download model weights:
- Qwen2-VL-2B: ~2GB (older version)
- Qwen2.5-VL-3B-Instruct: ~3GB (recommended, newest)
- Qwen2.5-VL-7B-Instruct: ~7GB (better accuracy)

### 2. Test oLmOCR

```bash
# Test with a trailer image
python3 test_olmocr.py test_trailer.jpg

# Or test with your own image
python3 test_olmocr.py path/to/your/image.jpg
```

### 3. Use in Your Application

oLmOCR follows the same interface as EasyOCR, so it can be used as a drop-in replacement:

```python
from app.ocr.olmocr_recognizer import OlmOCRRecognizer

# Initialize (use 3B model for edge devices - smallest in Qwen2.5-VL series)
ocr = OlmOCRRecognizer(
    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    use_gpu=True
)

# Alternative: Use older 2B model if memory is very limited
# ocr = OlmOCRRecognizer(model_name="Qwen/Qwen2-VL-2B", use_gpu=True)

# Recognize text (same interface as EasyOCR)
result = ocr.recognize(image)
print(f"Text: {result['text']}, Confidence: {result['conf']}")
```

## Model Selection

### For Jetson Orin (Edge Device)

**Recommended: Qwen2.5-VL-3B-Instruct**
- Smallest model in Qwen2.5-VL series (~3GB)
- Good balance of accuracy and speed
- Works well with 8GB+ GPU memory

```python
ocr = OlmOCRRecognizer(model_name="Qwen/Qwen2.5-VL-3B-Instruct")
```

**Alternative: Qwen2-VL-2B** (if memory is very limited)
- Older version but smaller (~2GB)
- Slightly less accurate than 3B

```python
ocr = OlmOCRRecognizer(model_name="Qwen/Qwen2-VL-2B")
```

### For Better Accuracy (if you have resources)

**Qwen2.5-VL-3B-Instruct**
- Better accuracy than 2B
- Requires ~4GB GPU memory
- Slower inference

**Qwen2.5-VL-7B-Instruct**
- Best accuracy
- Requires ~8GB+ GPU memory
- Slowest inference

## Performance Comparison

| Model | Size | GPU Memory | Speed | Accuracy |
|-------|------|------------|-------|----------|
| Qwen2-VL-2B | ~2GB | ~4GB | Fast | Good |
| Qwen2.5-VL-3B-Instruct | ~3GB | ~6GB | Medium | Better |
| Qwen2.5-VL-7B-Instruct | ~7GB | ~10GB+ | Slow | Best |

## Vertical Text Recognition

oLmOCR (Qwen2.5-VL) handles vertical text naturally. The model is trained to:
- Recognize text in any orientation
- Read vertical text from top to bottom
- Handle mixed horizontal/vertical layouts

**Example:**
```python
# oLmOCR automatically handles vertical text
result = ocr.recognize(vertical_text_image)
# No need for rotation preprocessing!
```

## Integration with Existing Pipeline

To use oLmOCR in your main application, update `app/main_trt_demo.py`:

```python
# Add oLmOCR as an option
from app.ocr.olmocr_recognizer import OlmOCRRecognizer

# Initialize oLmOCR (fallback or primary)
try:
    self.ocr = OlmOCRRecognizer(
        model_name="Qwen/Qwen2.5-VL-2B-Instruct",
        use_gpu=True
    )
    print("✓ Using oLmOCR (Qwen2.5-VL)")
except Exception as e:
    print(f"oLmOCR not available: {e}")
    # Fallback to EasyOCR or PaddleOCR
```

## Troubleshooting

### Issue: "Out of memory"

**Solution:**
1. Use Qwen2-VL-2B instead of Qwen2.5-VL-3B/7B
2. Or use the 3B model with CPU mode
3. Reduce batch size (process one image at a time)
4. Use CPU mode (slower but uses less GPU memory):
   ```python
   ocr = OlmOCRRecognizer(model_name="Qwen/Qwen2-VL-2B", use_gpu=False)
   ```

### Issue: "Model download failed"

**Solution:**
1. Check internet connection
2. Set HuggingFace cache directory:
   ```bash
   export HF_HOME=/path/to/cache
   ```
3. Manually download model from HuggingFace

### Issue: Slow inference

**Solution:**
1. Use smaller model (2B instead of 7B)
2. Enable GPU acceleration (`use_gpu=True`)
3. Consider TensorRT optimization (advanced)

### Issue: Poor accuracy on vertical text

**Solution:**
1. Try larger model (7B or 32B)
2. Ensure image preprocessing is appropriate
3. Check image quality (resolution, contrast)
4. Make sure you're using Qwen2.5-VL (newer) not Qwen2-VL (older)

## Comparison with EasyOCR

| Feature | EasyOCR | oLmOCR (Qwen2.5-VL) |
|---------|---------|---------------------|
| Vertical Text | Good | Excellent |
| Horizontal Text | Excellent | Excellent |
| Speed | Fast | Medium-Slow |
| Resource Usage | Low | High |
| Model Size | ~500MB | ~2-7GB |
| Edge Device | ✅ Good | ⚠️ Requires resources |
| Accuracy | High | Very High |

## Recommendations

**For your use case (vertical text on Jetson Orin):**

1. **Try oLmOCR with 3B model first** - Good balance for edge devices (Qwen2.5-VL-3B-Instruct)
2. **If memory is limited**, try Qwen2-VL-2B (older but smaller)
3. **If accuracy is insufficient**, try 7B model (if GPU memory allows)
4. **If speed is critical**, stick with EasyOCR and improve preprocessing
5. **Consider hybrid approach**: Use EasyOCR for horizontal text, oLmOCR for vertical text

## Alternative: MMOCR

If oLmOCR is too resource-intensive, consider **MMOCR** which also handles vertical text well and is more optimized for edge devices:

```bash
pip3 install mmocr
```

See `OCR_RECOMMENDATIONS.md` for MMOCR setup.

## References

- [Qwen2.5-VL GitHub](https://github.com/QwenLM/Qwen2-VL)
- [HuggingFace Qwen2.5-VL Models](https://huggingface.co/Qwen)
- [oLmOCR (Allen AI)](https://github.com/allenai/olmocr)




