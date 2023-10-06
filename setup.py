from setuptools import setup, find_packages

print(find_packages())
setup(
    name='sense',
    version='0.0.0.1',
    packages=find_packages(),
    install_requires=[
        "openai",
        "python-dotenv",
    ]
)
