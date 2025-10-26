"""Setup configuration for strands-agentcore-tools."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = (
    readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""
)

setup(
    name="strands-agentcore-tools",
    version="0.1.1",
    description="Strands tools for AWS Bedrock AgentCore lifecycle management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Cagatay Cali",
    author_email="cagataycali@icloud.com",
    url="https://github.com/cagataycali/strands-agentcore-tools",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=[
        "bedrock-agentcore",
        "strands-agents",
        "strands-agents-tools",
        "pyyaml",
        "boto3",
    ],
    extras_require={
        "dev": [
            "mypy>=1.8.0",
            "black>=24.0.0",
            "types-PyYAML",
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="aws bedrock agentcore strands ai agents tools",
    project_urls={
        "Bug Reports": "https://github.com/cagataycali/strands-agentcore-tools/issues",
        "Source": "https://github.com/cagataycali/strands-agentcore-tools",
    },
)
