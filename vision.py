"""What the caller's camera shows, read on this Mac by Apple's Vision framework: the text in
the picture, what kind of scene it is, how many faces. No cloud, no model to download.
It does not write descriptions: it reads and labels."""
import logging

import numpy as np

log = logging.getLogger("vision")

try:
    import Quartz
    import Vision
    from Foundation import NSURL
    READY = True
except ImportError:                 # not macOS, or pyobjc-framework-Vision is not installed
    READY = False


def rgb(i420, width, height, turns=0):
    """A raw call frame (I420: a full-size brightness plane, two quarter-size colour planes) as RGB."""
    y_size, c_size = width * height, (width // 2) * (height // 2)
    if len(i420) < y_size + 2 * c_size:
        raise ValueError(f"frame of {len(i420)} bytes is not {width}x{height} I420")
    raw = np.frombuffer(i420, dtype=np.uint8)
    y = raw[:y_size].reshape(height, width).astype(np.float32)
    u = raw[y_size:y_size + c_size].reshape(height // 2, width // 2).astype(np.float32) - 128
    v = raw[y_size + c_size:y_size + 2 * c_size].reshape(height // 2, width // 2).astype(np.float32) - 128
    u = u.repeat(2, 0).repeat(2, 1)
    v = v.repeat(2, 0).repeat(2, 1)
    y = y[:u.shape[0], :u.shape[1]]
    out = np.stack([y + 1.402 * v, y - 0.344 * u - 0.714 * v, y + 1.772 * u], -1)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return np.rot90(out, -turns) if turns else out      # the phone held sideways


def thumb(i420, width, height):
    """A tiny brightness picture: two frames are compared through it."""
    y = np.frombuffer(i420, dtype=np.uint8, count=width * height).reshape(height, width)
    return y[::max(1, height // 18), ::max(1, width // 32)][:18, :32].astype(np.int16)


def changed(a, b, by=6.0):
    return a is None or b is None or a.shape != b.shape or float(np.abs(a - b).mean()) > by


def cgimage(pixels):
    h, w = pixels.shape[:2]
    data = np.dstack([pixels, np.full((h, w, 1), 255, np.uint8)]).tobytes()
    provider = Quartz.CGDataProviderCreateWithData(None, data, len(data), None)
    return Quartz.CGImageCreate(w, h, 8, 32, w * 4, Quartz.CGColorSpaceCreateDeviceRGB(),
                                Quartz.kCGImageAlphaNoneSkipLast, provider, None, False,
                                Quartz.kCGRenderingIntentDefault)


def save(image, path):
    dest = Quartz.CGImageDestinationCreateWithURL(NSURL.fileURLWithPath_(str(path)), "public.jpeg", 1, None)
    Quartz.CGImageDestinationAddImage(dest, image, {Quartz.kCGImageDestinationLossyCompressionQuality: 0.7})
    Quartz.CGImageDestinationFinalize(dest)


def load(path):
    src = Quartz.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(str(path)), None)
    return Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)


def read(image, languages=("ru", "en"), max_text=700):
    """-> "a computer screen, a document; 1 face; text: …" (empty if there is nothing to tell)."""
    text = Vision.VNRecognizeTextRequest.alloc().init()
    text.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    text.setRecognitionLanguages_([{"ru": "ru-RU", "en": "en-US", "uk": "uk-UA", "de": "de-DE", "es": "es-ES",
                                    "fr": "fr-FR", "it": "it-IT", "pt": "pt-BR"}.get(c[:2], "en-US") for c in languages])
    text.setUsesLanguageCorrection_(True)
    kinds = Vision.VNClassifyImageRequest.alloc().init()
    faces = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    ok, error = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None) \
        .performRequests_error_([text, kinds, faces], None)
    if not ok:
        raise RuntimeError(f"Vision could not read the frame: {error}")
    labels = [o.identifier().replace("_", " ") for o in (kinds.results() or []) if o.confidence() > 0.35][:6]
    lines = []
    for o in text.results() or []:
        best = o.topCandidates_(1)
        if best and best[0].confidence() > 0.4 and len(best[0].string().strip()) > 1:
            lines.append(best[0].string().strip())
    words = " / ".join(lines)
    if len(words) > max_text:
        words = words[:max_text] + " …"
    n = len(faces.results() or [])
    parts = [", ".join(labels)] if labels else []
    if n:
        parts.append(f"{n} face{'s' if n > 1 else ''}")
    if words:
        parts.append("text in the picture: " + words)
    return "; ".join(parts)


def describe(i420, width, height, turns=0, languages=("ru", "en"), keep=None):
    image = cgimage(np.ascontiguousarray(rgb(i420, width, height, turns)))
    if keep:
        try:
            save(image, keep)
        except Exception:
            log.exception("could not save the frame")
    return read(image, languages)
