from setuptools import find_packages, setup


setup(
    name="my-video-inpainting-model",
    version="0.1.0",
    description="Simple HRNet-based video inpainting detector",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "opencv-python",
        "albumentations",
        "PyYAML",
    ],
)
