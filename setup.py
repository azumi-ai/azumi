from setuptools import setup, find_packages

setup(
    name="azumi",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.5.0",
        "fastapi>=0.65.0",
        "uvicorn>=0.13.0",
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "pydantic>=1.8.0",
        "sqlalchemy>=1.4.0",
        "python-multipart>=0.0.5",
        "python-jose>=3.3.0",
        "passlib>=1.7.4",
        "bcrypt>=3.2.0",
        "cryptography>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "isort>=5.8.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ]
    },
    author="Azumi Team",
    author_email="support@azumi.fun",
    description="A framework for creating artificial personalities and narrative ecosystems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/azumi-ai/azumi",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
)
