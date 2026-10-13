# sps-genai — Applied Generative AI

FastAPI service with three models:

- **Bigram text generation** (Module 3 class activity) — `POST /generate`
- **Word embeddings with spaCy** (Assignment 1) — `POST /embedding`
  Returns the 300-dimensional `en_core_web_lg` vector for a word.
- **CNN image classifier** (Assignment 2) — `POST /classify-image`
  Returns the most likely CIFAR-10 classes for an uploaded image.

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs

## Run locally (without Docker)

```bash
uv sync
uv run fastapi dev app/main.py
```

## Example requests

```bash
# Word embedding
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"word": "apple"}'

# Text generation
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"start_word": "the", "length": 10}'

# Image classification (top 3 classes)
curl -X POST "http://127.0.0.1:8000/classify-image?top_k=3" \
  -F "file=@cat.jpg;type=image/jpeg"
```

`/embedding` returns `404` if the word is not in the model's vocabulary,
and `422` if the word is empty.

`/classify-image` accepts JPEG or PNG uploads. It returns `415` for other
file types, `400` for empty or unreadable images, and `422` if `top_k`
is outside 1–10.

## Image classifier (Assignment 2)

The CNN in `app/cnn_model.py` follows the assignment spec: 64×64×3 input,
two Conv2D (16 and 32 filters, 3×3, stride 1, padding 1) + ReLU + 2×2 max
pooling blocks, then fully connected layers of 100 and 10 units
(825,398 parameters). CIFAR-10 images (32×32) are resized to 64×64 to
match the spec input.

Trained for 10 epochs on CIFAR-10 (Adam, lr 0.001, batch size 64) with
`python -m training.train_cnn --epochs 10`, run on a Google Colab GPU.
Trained weights: `app/weights/cifar10_cnn.pt`.

```
epoch     loss  train acc  test acc
    1   1.3666      51.0%     59.5%
    2   0.9978      64.8%     65.8%
    3   0.8294      70.6%     66.2%
    4   0.7020      75.2%     67.8%
    5   0.5887      79.3%     67.9%
    6   0.4750      83.3%     67.3%
    7   0.3757      87.0%     67.0%
    8   0.2876      90.0%     67.0%
    9   0.2125      92.6%     66.0%
   10   0.1674      94.2%     65.1%
Final test accuracy: 65.07%
```

Test accuracy peaked at 67.9% after epoch 5 and then declined, while
training accuracy kept rising to 94.2%. After epoch 5 the model began
overfitting, memorizing training images rather than learning patterns that
generalize. The spec's architecture has no regularization (such as Dropout
or BatchNorm), so this is expected; stopping around epoch 5 would have given
slightly better test accuracy.

Because the model learned from small 32×32 CIFAR-10 images, predictions on
real high-resolution photos are less reliable than the test accuracy suggests.