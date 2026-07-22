from . import text, pdf, office, image  # noqa: F401
# audio and video import heavy deps (faster-whisper, scenedetect) lazily inside
# their functions, so importing the package stays cheap.
