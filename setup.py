from setuptools import setup, find_packages

# Читаем длинное описание из README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Читаем зависимости из requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="microstrip-defect-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Моделирование и классификация дефектов микрополосковой линии",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/microstrip-defect-classifier",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "mpl-generate-data = scripts.generate_data:main",
            "mpl-train-model = scripts.train_model:main",
            "mpl-evaluate = scripts.evaluate:main",
            "mpl-visualize = scripts.visualize_data:main",
        ],
    },
    include_package_data=True,
)