from setuptools import setup, find_packages

setup(
    name="diffusion_policy",
    packages=[
        package for package in find_packages() 
        if package.startswith("diffusion_policy")
    ],
    python_requires=">=3.10",
    setup_requires=["setuptools>=62.3.0"],
    include_package_data=True,
    install_requires=[
        "diffusers==0.27.2",
        "einops==0.8.0",
        "torch==2.2.0",
        "torchvision==0.17.0",
    ]
)
