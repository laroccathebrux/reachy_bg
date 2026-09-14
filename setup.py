"""Setup script for Eldritch Horror Reachy Agent"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="eldritch-horror-reachy",
    version="0.1.0",
    description="Embodied Conversational AI Agent for Eldritch Horror Board Game",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alessandro La Rocca Silveira",
    author_email="alessandro@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "reachy-sdk>=1.0.0",
        "ollama>=0.1.0",
        "qdrant-client>=2.7.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "segment-anything>=1.0",
        "sentence-transformers>=2.2.2",
        "openai-whisper>=20230314",
        "elevenlabs>=0.2.0",
        "transformers>=4.34.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
